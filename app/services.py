import os
import json
import re
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import logging
import urllib3
import time
from typing import Any, Generator
from pathlib import Path
import sqlite3
import threading
from pydantic import BaseModel, Field
from typing import List, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

# LangChain Imports
from ddgs import DDGS
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.exceptions import OutputParserException


# Pydantic Models for structured output
class QueryAssessment(BaseModel):
    assessment: Literal["specific_topic", "broad_concept"] = Field(
        description="評估使用者問題是 'specific_topic' (具體主題) 還是 'broad_concept' (廣泛概念)。"
    )


class QueryClarification(BaseModel):
    intro_sentence: str = Field(description="引導使用者選擇的開場白。")
    options: List[str] = Field(
        description="提供給使用者的 3 到 4 個具體研究方向選項，最後一個選項必須是『以上全部，為我生成一份全面的綜合報告』。")


class TaskDecomposition(BaseModel):
    sub_tasks: List[str] = Field(
        description="將複雜研究目標分解為 2 到 4 個清晰、可執行的研究子問題列表。")


class SearchStrategy(BaseModel):
    search_queries: List[str] = Field(
        description="針對研究子任務生成的 3 到 5 個具體、多樣化的搜尋引擎關鍵字列表。")


class SearchResult(BaseModel):
    id: int
    title: str
    url: str


class AssessedSearchResult(BaseModel):
    id: int
    is_relevant: bool


class SearchResultAssessmentList(BaseModel):
    results: List[AssessedSearchResult]


class ContentRelevanceResult(BaseModel):
    is_truly_relevant: bool
    summary: str = Field(
        description="A brief summary of the content if relevant, or a reason for irrelevance.")


class ClaimList(BaseModel):
    claims: List[str] = Field(
        description="A list of verifiable, factual claims extracted from a report.")


class FactCheckResult(BaseModel):
    is_verified: bool = Field(
        description="Whether the statement is verified by the source material.")
    supporting_evidence: str = Field(
        description="The snippet from the source material that supports the statement, or 'null' if not verifiable.")


class Chapter(BaseModel):
    title: str = Field(description="章節標題，必須使用繁體中文。")
    description: str = Field(description="章節內容的簡要描述，如果包含表格數據，請註明『使用表格呈現』。")
    key_points: List[str] = Field(description="一個包含本章節需要闡述的 3 到 5 個核心要點的列表。")


class ReportBlueprint(BaseModel):
    title: str = Field(description="報告的總標題，必須使用繁體中文。")
    chapters: List[Chapter] = Field(description="報告的章節列表，每個章節包含標題和描述。")


try:
    from playwright.sync_api import sync_playwright, Browser
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning(
        "Playwright 未安裝。爬蟲功能將僅使用 requests。建議執行 'pip install playwright && playwright install' 來增強爬蟲能力。")

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF 未安裝。PDF 處理功能將不可用。建議執行 'pip install PyMuPDF'。")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ClientDisconnectedError(Exception):
    """Custom exception for client disconnects."""
    pass


def get_ollama_models(ollama_base_url="http://localhost:11434"):
    try:
        response = requests.get(f"{ollama_base_url}/api/tags")
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]
    except Exception as e:
        logging.error(f"存取 Ollama 模型時發生錯誤: {e}")
        return []


class ConversationalRAG:
    def __init__(self, config: dict):
        self.config = config
        self.use_web_search = True
        self.ollama_base_url = self.config["OLLAMA_BASE_URL"]
        self.prompts = {}
        self.active_tasks = {}
        self.blueprint_cache = {}
        self.db_path = "database.db"
        self._init_database()

        logging.info("正在設置 LLM...")
        try:
            self.llm = OllamaLLM(
                model=self.config["llm_model"],
                base_url=self.ollama_base_url,
                request_timeout=self.config.get('OLLAMA_REQUEST_TIMEOUT')
            )
            self.llm.invoke("Hi", stop=["Hi"])
            self.current_llm_model = self.config["llm_model"]
        except Exception as e:
            logging.error(f"設置 LLM 失敗: {e}")
            raise

        self._init_prompts()
        logging.info("✅ 系統已就緒 (使用 DDGS 作為搜尋引擎)。")

    def close(self):
        pass

    def _init_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    report TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
            logging.info(f"✅ 資料庫 '{self.db_path}' 已成功初始化。")
        except Exception as e:
            logging.error(f"❌ 初始化資料庫失敗: {e}")
            raise

    def add_history_entry(self, title: str, report: str) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO history (title, report) VALUES (?, ?)", (title, report))
        entry_id = cursor.lastrowid
        conn.commit()
        conn.close()
        logging.info(f"💾 已將報告 '{title[:20]}...' 儲存至歷史紀錄 (ID: {entry_id})。")
        return entry_id

    def get_history_list(self) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, title, timestamp FROM history ORDER BY timestamp DESC")
        entries = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return entries

    def get_history_entry(self, entry_id: int) -> dict:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history WHERE id = ?", (entry_id,))
        entry = cursor.fetchone()
        conn.close()
        return dict(entry) if entry else None

    def delete_history_entry(self, entry_id: int) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history WHERE id = ?", (entry_id,))
        conn.commit()
        deleted_rows = cursor.rowcount
        conn.close()
        if deleted_rows > 0:
            logging.info(f"🗑️ 已從歷史紀錄中刪除 ID: {entry_id}。")
            return True
        return False

    def _load_all_prompts(self, directory: str) -> dict:
        prompts = {}
        base_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_dir = os.path.join(base_dir, directory)
        if not os.path.isdir(prompts_dir):
            return {}
        logging.info(f"🔍 正在從 '{prompts_dir}' 加載 Prompt 模板...")
        for filename in os.listdir(prompts_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(prompts_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    prompt_name = os.path.splitext(filename)[0]
                    prompts[prompt_name] = PromptTemplate.from_template(
                        f.read(), template_format="f-string")
                    logging.info(f"   -> 已加載: {prompt_name}")
        return prompts

    def _init_prompts(self):
        base_prompts = self._load_all_prompts("prompts")
        self.prompts = {}

        self.parsers = {
            "query_assessor": PydanticOutputParser(pydantic_object=QueryAssessment),
            "query_clarifier": PydanticOutputParser(pydantic_object=QueryClarification),
            "task_decomposition": PydanticOutputParser(pydantic_object=TaskDecomposition),
            "search_strategist": PydanticOutputParser(pydantic_object=SearchStrategy),
            "answer_blueprint_generator": PydanticOutputParser(pydantic_object=ReportBlueprint),
            "search_result_assessor": PydanticOutputParser(pydantic_object=SearchResultAssessmentList),
            "content_relevance_filter": PydanticOutputParser(pydantic_object=ContentRelevanceResult),
            "claim_extractor": PydanticOutputParser(pydantic_object=ClaimList),
            "fact_checker": PydanticOutputParser(pydantic_object=FactCheckResult),
        }

        for prompt_name, parser in self.parsers.items():
            if prompt_name in base_prompts:
                template_string = base_prompts[prompt_name].template
                self.prompts[prompt_name] = PromptTemplate(
                    template=template_string + "\n{format_instructions}",
                    input_variables=base_prompts[prompt_name].input_variables,
                    partial_variables={
                        "format_instructions": parser.get_format_instructions()},
                    template_format="f-string"
                )
                logging.info(f"   -> 已為 {prompt_name} 注入 Pydantic 格式指令。")
            else:
                logging.warning(f"⚠️ 警告: 找不到 {prompt_name} 的基礎 Prompt 模板。")

        for prompt_name, prompt_template in base_prompts.items():
            if prompt_name not in self.prompts:
                self.prompts[prompt_name] = prompt_template
                logging.info(f"   -> 已加載非 Pydantic Prompt: {prompt_name}")

    def set_llm_model(self, model_name: str):
        logging.info(f"\n🔄 正在切換 LLM 模型至: {model_name}")
        try:
            self.llm = OllamaLLM(
                model=model_name, base_url=self.ollama_base_url)
            self.llm.invoke("Hi", stop=["Hi"])
            self.current_llm_model = model_name
            logging.info(f"✅ LLM 模型成功切換為: {self.current_llm_model}")
            return True
        except Exception as e:
            logging.error(f"切換 LLM 模型失敗: {e}")
            return False

    def set_web_search(self, enabled: bool):
        logging.info(f"🔄 將網路搜尋設置為: {'啟用' if enabled else '停用'}")
        self.use_web_search = enabled
        return True

    def _agent_based_web_search(self, question: str, task: str):
        logging.info(f"🤖 (Agent模式) 啟動 DDGS 通用網路搜尋: '{question}'")
        all_source_docs = []

        assessor_chain = self.prompts.get(
            "search_result_assessor") | self.llm | self.parsers["search_result_assessor"]
        content_filter_chain = self.prompts.get(
            "content_relevance_filter") | self.llm | self.parsers["content_relevance_filter"]

        DOMAIN_BLACKLIST = ["zhihu.com", "facebook.com",
                            "instagram.com", "linkedin.com", "medium.com"]

        try:
            with DDGS() as ddgs:
                search_results = list(
                    ddgs.text(question, max_results=30, region="tw-zh"))

            if not search_results:
                logging.warning(
                    f"   -> DDGS for query '{question}' returned no results.")
                return []

            logging.info(
                f"   -> DDGS for query '{question}' returned {len(search_results)} results. Now performing Layer 1 (Batch Relevance Scan)...")

            filtered_results = [res for res in search_results if "href" in res and res.get(
                "title") and not any(blacklisted in res["href"] for blacklisted in DOMAIN_BLACKLIST)]
            input_list = [SearchResult(id=i, title=res["title"], url=res["href"])
                          for i, res in enumerate(filtered_results)]

            try:
                assessment_list_obj = assessor_chain.invoke(
                    {"task": task, "search_results_json": json.dumps([r.dict() for r in input_list])})
                relevant_ids = {
                    res.id for res in assessment_list_obj.results if res.is_relevant}
                urls_to_browse = [
                    res.url for res in input_list if res.id in relevant_ids]
                logging.info(
                    f"   -> Layer 1 Scan complete. {len(urls_to_browse)} URLs passed initial relevance filter.")
            except Exception as e:
                logging.error(
                    f"   -> ⚠️ Layer 1 (Batch Relevance Scan) failed: {e}. Skipping filtering.")
                urls_to_browse = [res.url for res in input_list][:10]

            def scrape_and_verify_url(url):
                try:
                    with sync_playwright() as p:
                        browser = p.chromium.launch(headless=True)
                        try:
                            content = self._scrape_webpage_text(url, browser)
                            if "無法爬取" in content or len(content) < 100:
                                logging.warning(
                                    f"   -> Skipping {url} due to insufficient content.")
                                return None

                            filter_result = content_filter_chain.invoke(
                                {"task": task, "content": content})
                            if filter_result.is_truly_relevant:
                                logging.info(f"   -> ✅ DEEPLY RELEVANT: {url}")
                                return Document(page_content=content, metadata={"source": f"網頁瀏覽: {url}"})
                            else:
                                logging.info(
                                    f"   -> ❌ CONTENT IRRELEVANT: {url}. Reason: {filter_result.summary}")
                                return None
                        finally:
                            browser.close()
                except Exception as e:
                    logging.error(
                        f"   -> ⚠️ Error during scrape_and_verify_url for {url}: {e}")
                    return None

            logging.info(
                f"   -> Now performing Layer 2 (Deep Content Verification) on {len(urls_to_browse)} URLs in parallel...")
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {executor.submit(
                    scrape_and_verify_url, url): url for url in urls_to_browse}
                for future in as_completed(future_to_url):
                    result_doc = future.result()
                    if result_doc:
                        all_source_docs.append(result_doc)

            logging.info(
                f"   -> Agent 搜尋 '{question}' 完成，共找到 {len(all_source_docs)} 份高品質文件。")
            return all_source_docs
        except Exception as e:
            logging.error(f"❌ 在 Agent 搜尋過程中發生錯誤: {e}", exc_info=True)
            return []

    def _scrape_webpage_text(self, url: str, browser: Browser):
        if url.lower().endswith('.pdf'):
            if not PYMUPDF_AVAILABLE:
                return "無法處理 PDF 文件，因為 PyMuPDF 未安裝。"
            try:
                logging.info(f"📄 (PyMuPDF模式) 正在處理 PDF 網址: {url}")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'}
                response = requests.get(
                    url, headers=headers, timeout=30, verify=False)
                response.raise_for_status()
                with fitz.open(stream=response.content, filetype="pdf") as pdf_document:
                    text = "".join(page.get_text() for page in pdf_document)
                logging.info(f"✅ (PyMuPDF) 成功提取 PDF 文字，長度: {len(text)} 字元。")
                return text
            except Exception as e:
                logging.error(f"❌ (PyMuPDF) 處理 PDF 時失敗: {e}")
                return f"無法爬取 PDF，錯誤: {e}"

        logging.info(f"🕸️ (Playwright模式) 正在嘗試爬取網址: {url}")
        page = None
        try:
            if any(url.lower().endswith(ext) for ext in ['.doc', '.docx', '.zip', '.rar', '.xls', '.xlsx']):
                return "無法爬取網站，錯誤: 不支持的文件類型"
            page = browser.new_page()
            page.goto(url, timeout=30000, wait_until='domcontentloaded')
            try:
                page.wait_for_selector(
                    "main, article, #content, #main-content, .post-content, .article-body", timeout=10000)
                logging.info("   -> ✅ 成功等到關鍵內容區塊。")
            except Exception as wait_e:
                logging.warning(f"   -> ⚠️ 未能等到特定內容區塊，將直接抓取現有內容。錯誤: {wait_e}")
            html_content = page.content()
            soup = BeautifulSoup(html_content, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "aside", "header", "iframe", "form"]):
                element.decompose()
            text = "\n".join(line.strip()
                             for line in soup.get_text().splitlines() if line.strip())
            if len(text) < 300:
                logging.warning(
                    f"⚠️ (Playwright) 从 {url} 提取的有效文字過少 ({len(text)} 字元)，可能不是主要內容。")
            logging.info(f"✅ (Playwright) 成功獲取網頁文字，長度: {len(text)} 字元。")
            return text
        except Exception as e:
            logging.warning(f"❌ (Playwright) 失敗: {e}。將回退至 Requests 模式。")
            return "無法爬取網站，錯誤: Playwright 連接失敗"
        finally:
            if page and not page.is_closed():
                page.close()

    def _perform_final_cleanup(self, task_id: str):
        if task_id and task_id in self.active_tasks:
            logging.info(f"🧹 任務 {task_id} 正在最終註銷。")
            del self.active_tasks[task_id]

    def _send_done_signal(self) -> Generator[str, None, None]:
        yield f"data: {json.dumps({'type': 'done', 'content': '[DONE]'})}\n\n"
        logging.info("✅ 資料流已正常關閉。")

    def _handle_clarification(self, question: str, task_id: str) -> Generator[str, None, None]:
        logging.info(f"💬 KAIZEN 路由：問題過於寬泛，啟動問題澄清流程。")
        clarifier_template = self.prompts.get("query_clarifier")
        if not clarifier_template:
            raise ValueError("query_clarifier.txt 模板未找到！")

        clarifier_chain = clarifier_template | self.llm | self.parsers["query_clarifier"]
        clarification_obj = clarifier_chain.invoke({"question": question})

        clarification_json = clarification_obj.dict()
        yield f"data: {json.dumps({'type': 'clarification', 'data': clarification_json})}\n\n"

    def _stream_research_and_blueprint(self, question: str, stream: bool, task_id: str) -> Generator[str, None, None]:
        logging.info(f"🚀 啟動 [第一階段] 專家小組工作流 (Task ID: {task_id})...")
        start_time = time.time()

        def send_event(event_data):
            try:
                yield event_data
            except GeneratorExit:
                raise ClientDisconnectedError(
                    f"Client disconnected during task {task_id}")

        try:
            # --- 心跳機制初始化 ---
            last_heartbeat = time.time()
            HEARTBEAT_INTERVAL = 15  # seconds

            def yield_heartbeat_if_needed():
                nonlocal last_heartbeat
                if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
                    # SSE 註解格式，客戶端 JS 會忽略它，但能保持 TCP 連接
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    last_heartbeat = time.time()
                    logging.debug(f"Sent heartbeat for task {task_id}")
            # --- 心跳機制初始化結束 ---

            def check_cancellation():
                if task_id and self.active_tasks.get(task_id, {}).get("is_cancelled"):
                    raise InterruptedError(
                        f"Task {task_id} cancelled by user.")

            check_cancellation()
            yield from send_event(f"data: {json.dumps({'type': 'status', 'message': '步驟 1/3: 正在拆解研究任務...'})}\n\n")
            yield from yield_heartbeat_if_needed() # <<-- 加入心跳檢查點

            task_decomp_template = self.prompts.get("task_decomposition")
            task_decomp_chain = task_decomp_template | self.llm | self.parsers[
                "task_decomposition"]
            task_decomp_obj = task_decomp_chain.invoke({"question": question})
            sub_tasks = list(dict.fromkeys(
                [t.strip() for t in task_decomp_obj.sub_tasks]))[:4]
            logging.info(f"✅ 清理與驗證後的子任務 ({len(sub_tasks)} 條): {sub_tasks}")

            research_memos = []
            all_source_documents = []
            analyst_template = self.prompts.get("research_synthesizer")
            search_strategist_template = self.prompts.get("search_strategist")
            if not all([analyst_template, search_strategist_template]):
                raise ValueError("一個或多個關鍵的 Prompt 模板未找到！")
            strategist_chain = search_strategist_template | self.llm | self.parsers[
                "search_strategist"]

            # 主要的循環，最適合加入心跳
            for i, task in enumerate(sub_tasks):
                check_cancellation()
                yield from yield_heartbeat_if_needed() # <<-- 每次循環開始時檢查
                yield from send_event(f"data: {json.dumps({'type': 'status', 'message': f'步驟 2.{i+1}/{len(sub_tasks)}: 正在綜合研究 "{task[:20]}..."'})}\n\n")

                search_strategy_obj = strategist_chain.invoke(
                    {"question": question, "task": task})
                search_queries = list(dict.fromkeys(
                    [q.strip() for q in search_strategy_obj.search_queries])) or [task]
                logging.info(
                    f"   -> 策略師為 '{task}' 生成了 {len(search_queries)} 個搜尋向量: {search_queries}")
                task_specific_docs = []
                
                # 內部循環，這裡的等待時間最長
                for query in search_queries:
                    check_cancellation()
                    yield from yield_heartbeat_if_needed() # <<-- 每次查詢前檢查
                    docs_for_query = self._agent_based_web_search(query, task) # <== 長時間操作
                    if docs_for_query:
                        task_specific_docs.extend(docs_for_query)
                
                context = "注意：未能從網路找到相關資料。"
                if task_specific_docs:
                    all_source_documents.extend(task_specific_docs)
                    unique_contents, unique_docs_for_synthesis = set(), []
                    for doc in task_specific_docs:
                        if doc.page_content not in unique_contents:
                            unique_contents.add(doc.page_content)
                            unique_docs_for_synthesis.append(doc)
                    logging.info(
                        f"   -> 為子任務 '{task}' 匯總了 {len(unique_docs_for_synthesis)} 份不重複的文件進行綜合分析。")
                    context = "\n---\n".join(
                        [f"來源：{doc.metadata.get('source', '網路')}\n內容:\n{doc.page_content}" for doc in unique_docs_for_synthesis])
                else:
                    logging.warning(f"   -> 未能為子任務 '{task}' 找到任何網路資料。")
                
                yield from yield_heartbeat_if_needed() # <<-- LLM調用前檢查
                detailed_memo = self.llm.invoke(
                    analyst_template.format(context=context, question=task))

                research_memos.append(f"### 研究主題: {task}\n\n{detailed_memo}")
                logging.info(f"   -> 已為 '{task}' 生成詳細研究備忘錄。")

            final_context = "\n\n---\n\n".join(research_memos)

            check_cancellation()
            yield from send_event(f"data: {json.dumps({'type': 'status', 'message': '步驟 3/3: 正在生成報告大綱...'})}\n\n")
            yield from yield_heartbeat_if_needed() # <<-- 最後一步前檢查
            blueprint_gen_template = self.prompts.get(
                "answer_blueprint_generator")
            blueprint_chain = blueprint_gen_template | self.llm | self.parsers[
                "answer_blueprint_generator"]
            blueprint_obj = blueprint_chain.invoke(
                {"context": final_context, "question": question})

            blueprint_json = blueprint_obj.dict()
            logging.info("✅ 成功解析回答大綱 JSON。")

            serializable_sources = [{'page_content': doc.page_content,
                                     'metadata': doc.metadata} for doc in all_source_documents]

            end_time = time.time()
            duration_seconds = end_time - start_time
            duration_str = str(timedelta(seconds=int(duration_seconds)))

            self.blueprint_cache[task_id] = {
                'question': question,
                'context': final_context,
                'blueprint': blueprint_json,
                'sources': serializable_sources
            }
            logging.info(f"✅ 已為任務 {task_id} 快取藍圖資料。")

            yield from send_event(f"data: {json.dumps({'type': 'blueprint_generated', 'data': {'blueprint': blueprint_json, 'duration': duration_str}})}\n\n")
            logging.info(f"✅ 已為任務 {task_id} 生成藍圖並發送至前端，等待使用者確認。")

        except ClientDisconnectedError as e:
            logging.warning(f"[WORKFLOW CANCELLATION] {e}")
            self.active_tasks[task_id]['is_cancelled'] = True

    def _fact_check_report(self, report_text: str, context: str) -> Generator[str, None, None]:
        logging.info("🕵️ 啟動事後事實校驗流程...")
        yield f"data: {json.dumps({'type': 'status', 'message': '步驟 5/5: 正在進行事後事實校驗...'})}\n\n"

        try:
            claim_extractor_chain = self.prompts["claim_extractor"] | self.llm | self.parsers["claim_extractor"]
            fact_checker_chain = self.prompts["fact_checker"] | self.llm | self.parsers["fact_checker"]

            extracted_claims_obj = claim_extractor_chain.invoke(
                {"report_text": report_text})
            claims = extracted_claims_obj.claims
            logging.info(f"   -> 已從報告中提取 {len(claims)} 條關鍵論述進行校驗。")

            if not claims:
                yield f"data: {json.dumps({'type': 'content', 'content': '\n\n---\n\n## 🤖 自動事實校驗\n\n未能從報告中提取出可供校驗的關鍵論述。'})}\n\n"
                return

            fact_check_results = []
            for claim in claims:
                try:
                    check_result = fact_checker_chain.invoke(
                        {"context": context, "statement": claim})
                    fact_check_results.append((claim, check_result))
                except Exception as e:
                    logging.warning(f"校驗論述時出錯: '{claim}'. 錯誤: {e}")

            result_markdown = "\n\n---\n\n## 🤖 自動事實校驗\n\n本工具已自動校驗報告中的關鍵論述，結果如下：\n\n"
            for claim, result in fact_check_results:
                icon = "✅" if result.is_verified else "❌"
                evidence = result.supporting_evidence if result.is_verified else "未在原始資料中找到直接證據。"
                result_markdown += f"- **論述**: \"{claim}\"\n"
                result_markdown += f"  - **校驗結果**: {icon} {evidence}\n"

            yield f"data: {json.dumps({'type': 'content', 'content': result_markdown})}\n\n"
            logging.info("✅ 事實校驗流程完成。")

        except Exception as e:
            logging.error(f"❌ 在事實校驗流程中發生嚴重錯誤: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'content', 'content': '\n\n> **事實校驗失敗**: 在執行自動校驗時發生內部錯誤。'})}\n\n"

    def stream_write_report(self, task_id: str) -> Generator[str, None, None]:
        def check_cancellation():
            if task_id and self.active_tasks.get(task_id, {}).get("is_cancelled"):
                raise InterruptedError(f"Task {task_id} cancelled by user.")

        full_report_text = ""
        try:
            cached_data = self.blueprint_cache.get(task_id)
            if not cached_data:
                raise ValueError(f"無法在快取中找到任務 {task_id} 的藍圖資料。")

            # --- 心跳機制初始化 ---
            last_heartbeat = time.time()
            HEARTBEAT_INTERVAL = 15 # seconds

            def yield_heartbeat_if_needed():
                nonlocal last_heartbeat
                if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    last_heartbeat = time.time()
                    logging.debug(f"Sent heartbeat during report writing for task {task_id}")
            # --- 心跳機制初始化結束 ---

            question = cached_data['question']
            context = cached_data['context']
            blueprint = cached_data['blueprint']
            sources_json = cached_data['sources']

            all_source_documents = [Document(
                page_content=doc['page_content'], metadata=doc['metadata']) for doc in sources_json]

            check_cancellation()
            yield f"data: {json.dumps({'type': 'status', 'message': '步驟 4/4: 正在撰寫最終報告...'})}\n\n"
            chapter_writer_template = self.prompts.get("chapter_writer")
            if not chapter_writer_template:
                raise ValueError("chapter_writer.txt 模板未找到！")

            chapter_writer_chain = chapter_writer_template | self.llm | StrOutputParser()

            for chapter in blueprint.get('chapters', []):
                check_cancellation()
                yield from yield_heartbeat_if_needed() # <<-- 每個章節開始前檢查

                chapter_title = chapter.get('title', '無標題章節')
                # Yield chapter title as a header
                chapter_header = f"## {chapter_title}\n\n"
                full_report_text += chapter_header
                yield f"data: {json.dumps({'type': 'content', 'content': chapter_header})}\n\n"
                key_points_list = chapter.get('key_points', [])
                formatted_key_points = "\n".join(
                    f"- {point}" for point in key_points_list)

                for chunk in chapter_writer_chain.stream({
                    "question": question,
                    "context": context,
                    "chapter_title": chapter_title,
                    "key_points": formatted_key_points
                }):
                    check_cancellation()
                    # 重置心跳計時器，因為我們剛剛發送了真實數據
                    last_heartbeat = time.time() 
                    full_report_text += chunk
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

            logging.info("報告正文串流生成完畢！正在進行最終組裝...")
            unique_urls = {doc.metadata.get("source", "").replace("網頁瀏覽: ", "").strip(
            ) for doc in all_source_documents if doc.metadata.get("source", "").startswith("網頁瀏覽: ")}
            reference_list_str = "\n".join(
                [f"- {url}" for url in sorted(list(unique_urls)) if url])
            if reference_list_str:
                reference_section_str = f"\n\n---\n\n## 參考文獻\n{reference_list_str}"
                full_report_text += reference_section_str
                logging.info(f"✅ 已提取 {len(unique_urls)} 條獨特的參考文獻。")
                yield f"data: {json.dumps({'type': 'content', 'content': reference_section_str})}\n\n"

            # --- Fact-Checking Stage ---
            yield from yield_heartbeat_if_needed()
            yield from self._fact_check_report(full_report_text, context)

            yield f"data: {json.dumps({'type': 'status', 'message': '報告生成完畢！'})}\n\n"
            yield from self._send_done_signal()

        except InterruptedError:
            logging.info(f"🛑 報告撰寫任務 {task_id} 已被使用者成功中止。")
            yield f"data: {json.dumps({'type': 'status', 'message': '任務已取消。'})}\n\n"
            yield from self._send_done_signal()
        except Exception as e:
            logging.error(f"❌ 在報告撰寫流程中發生嚴重錯誤: {e}", exc_info=True)
            error_message = f'抱歉，執行報告撰寫時發生錯誤: {str(e)}'
            yield f"data: {json.dumps({'type': 'error', 'content': error_message})}\n\n"
            yield from self._send_done_signal()
        finally:
            if task_id in self.blueprint_cache:
                final_report_with_fact_check = self.blueprint_cache[task_id].get(
                    'final_report', full_report_text)
                if final_report_with_fact_check:
                    self.add_history_entry(
                        self.blueprint_cache[task_id]['question'], final_report_with_fact_check)
                self._perform_final_cleanup(task_id)

    def ask(self, question: str, stream: bool = True, task_id: str = None, bypass_assessment: bool = False) -> Generator[str, None, None]:
        if not task_id:
            task_id = f"task_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
        self.active_tasks[task_id] = {'is_cancelled': False}
        logging.info(f"\n🤔 收到請求並註冊任務 ID: {task_id}，問題: '{question}'")

        if not self.use_web_search:
            logging.info(" KAIZEN 路由：網路研究已停用，強制執行直接問答。 ")
            try:
                direct_answer_template = self.prompts.get("direct_answer")
                prompt = direct_answer_template.format(
                    question=question) if direct_answer_template else question
                for chunk in self.llm.stream(prompt):
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                yield from self._send_done_signal()
            finally:
                self._perform_final_cleanup(task_id)
            return

        is_complex_flow = False
        try:
            if bypass_assessment or re.search(r"https?://[\S]+", question):
                is_complex_flow = True
                yield from self._stream_research_and_blueprint(question, stream, task_id)
            else:
                assessor_template = self.prompts.get("query_assessor")
                assessor_chain = assessor_template | self.llm | self.parsers["query_assessor"]
                assessment_obj = assessor_chain.invoke({"question": question})
                assessment = assessment_obj.assessment

                if assessment == "specific_topic":
                    is_complex_flow = True
                    yield from self._stream_research_and_blueprint(question, stream, task_id)
                elif assessment == "broad_concept":
                    is_complex_flow = True
                    yield from self._handle_clarification(question, task_id)
                else:
                    is_complex_flow = True
                    yield from self._stream_research_and_blueprint(question, stream, task_id)

        except OutputParserException as e:
            logging.error(
                f"❌ 在 ASK 指揮官模式中發生 OutputParserException: {e}", exc_info=True)
            error_message = '報告生成失敗：AI 模型未能返回預期格式的數據，請嘗試調整問題或稍後重試。'
            yield f"data: {json.dumps({'type': 'error', 'content': error_message})}\n\n"
            yield from self._send_done_signal()
        except InterruptedError:
            logging.info(f" 任務 {task_id} 已被使用者成功中止。")
            yield f"data: {json.dumps({'type': 'status', 'message': '任務已取消。'})}\n\n"
            yield from self._send_done_signal()
        except Exception as e:
            logging.error(f"❌ 在 ASK 指揮官模式中發生嚴重錯誤: {e}", exc_info=True)
            error_message = f'抱歉，執行時發生了無法預期的錯誤: {str(e)}'
            yield f"data: {json.dumps({'type': 'error', 'content': error_message})}\n\n"
            yield from self._send_done_signal()
        
