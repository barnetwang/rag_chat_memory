import os
import json
import re
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import logging
import urllib3
import time
from typing import Any, Generator

# LangChain Imports
from ddgs import DDGS
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
        self.playwright = None
        self.browser = None
        self.config = config
        self.use_web_search = True
        self.ollama_base_url = self.config["OLLAMA_BASE_URL"]
        self.prompts = {}
        self.active_tasks = {}

        logging.info("正在設置 LLM...")
        try:
            self.llm = OllamaLLM(
                model=self.config["llm_model"], base_url=self.ollama_base_url)
            self.current_llm_model = self.config["llm_model"]
        except Exception as e:
            logging.error(f"設置 LLM 失敗: {e}")
            raise

        self._init_prompts()
        logging.info("✅ 系統已就緒 (使用 DDGS 作為搜尋引擎)。")

    def _init_playwright(self):
        if PLAYWRIGHT_AVAILABLE and not self.browser:
            try:
                logging.info("🌐 正在啟動共享的 Playwright 瀏覽器實例...")
                self.playwright = sync_playwright().start()
                self.browser = self.playwright.chromium.launch(headless=True)
                logging.info("✅ Playwright 共享瀏覽器實例已成功啟動。")
            except Exception as e:
                logging.error(f"❌ 啟動 Playwright 失敗: {e}")
                self.playwright = None
                self.browser = None

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
        self.prompts = self._load_all_prompts("prompts")

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

    def _agent_based_web_search(self, question: str):
        logging.info(f"🤖 (Agent模式) 啟動 DDGS 通用網路搜尋: '{question}'")
        all_source_docs = []
        if not PLAYWRIGHT_AVAILABLE:
            logging.warning("Playwright 未安裝，搜尋功能受限。")
            return []
        # --- 網站黑名單 ---
        DOMAIN_BLACKLIST = [
            "zhihu.com",      # 很難爬取，經常需要登入
            "facebook.com",   # 需要登入
            "instagram.com",  # 需要登入
            "linkedin.com",   # 需要登入
            "medium.com",     # 經常有付費牆或登入提示
        ]
        # ------------------------------------
        with sync_playwright() as playwright:
            browser: Browser = None
            try:
                browser = playwright.chromium.launch(headless=True)
                with DDGS() as ddgs:
                    search_results = ddgs.text(
                        question, max_results=20, region="tw-zh")
                if not search_results:
                    return []
                urls_to_browse = [
                    res["href"] for res in search_results
                    if "href" in res and not any(blacklisted in res["href"] for blacklisted in DOMAIN_BLACKLIST)
                ][:10]
                logging.info(f"   -> 已過濾掉黑名單網站，準備瀏覽以下 {len(urls_to_browse)} 個網頁...")
                if not urls_to_browse:
                    logging.warning("   -> 過濾後沒有可瀏覽的網頁。")
                    return []

                for url in urls_to_browse:
                    if not url:
                        continue
                    content = self._scrape_webpage_text(url, browser)
                    if "無法爬取" not in content and len(content) > 100:
                        all_source_docs.append(
                            Document(page_content=content, metadata={"source": f"網頁瀏覽: {url}"}))
                logging.info(f"   -> Agent 搜尋 '{question}' 完成，共找到 {len(all_source_docs)} 份有效文件。")
                return all_source_docs
            except Exception as e:
                logging.error(f"❌ 在 Agent 搜尋過程中發生錯誤: {e}", exc_info=True)
                return []
            finally:
                if browser:
                    browser.close()

    def _scrape_webpage_text(self, url: str, browser: Browser = None):
        if url.lower().endswith('.pdf'):
            if not PYMUPDF_AVAILABLE: return "無法處理 PDF 文件，因為 PyMuPDF 未安裝。"
            try:
                logging.info(f"📄 (PyMuPDF模式) 正在處理 PDF 網址: {url}")
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'}
                response = requests.get(url, headers=headers, timeout=30, verify=False)
                response.raise_for_status()
                with fitz.open(stream=response.content, filetype="pdf") as pdf_document:
                    text = "".join(page.get_text() for page in pdf_document)
                logging.info(f"✅ (PyMuPDF) 成功提取 PDF 文字，長度: {len(text)} 字元。")
                return text
            except Exception as e:
                logging.error(f"❌ (PyMuPDF) 處理 PDF 時失敗: {e}")
                return f"無法爬取 PDF，錯誤: {e}"
        if browser:
            logging.info(f"🕸️ (Playwright模式) 正在嘗試爬取網址: {url}")
            page = None
            try:
                if any(url.lower().endswith(ext) for ext in ['.doc', '.docx', '.zip', '.rar', '.xls', '.xlsx']): return "無法爬取網站，錯誤: 不支持的文件類型"
                page = browser.new_page()
                page.goto(url, timeout=30000, wait_until='domcontentloaded')
                try:
                    page.wait_for_selector("main, article, #content, #main-content, .post-content, .article-body", timeout=10000)
                    logging.info("   -> ✅ 成功等到關鍵內容區塊。")
                except Exception as wait_e:
                    logging.warning(f"   -> ⚠️ 未能等到特定內容區塊，將直接抓取現有內容。錯誤: {wait_e}")
                html_content = page.content()
                soup = BeautifulSoup(html_content, "html.parser")
                for element in soup(["script", "style", "nav", "footer", "aside", "header", "iframe", "form"]): element.decompose()
                text = "\n".join(line.strip() for line in soup.get_text().splitlines() if line.strip())
                if len(text) < 300: logging.warning(f"⚠️ (Playwright) 从 {url} 提取的有效文字过少 ({len(text)} 字元)，可能不是主要内容。")
                logging.info(f"✅ (Playwright) 成功獲取網頁文字，長度: {len(text)} 字元。")
                return text
            except Exception as e:
                logging.warning(f"❌ (Playwright) 失敗: {e}。將回退至 Requests 模式。")
                return "無法爬取網站，錯誤: Playwright 連接失敗"
            finally:
                if page and not page.is_closed(): page.close()
        logging.info(f"🕸️ (Requests模式) 正在嘗試爬取網址: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=30, verify=False)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "aside", "header", "iframe", "form"]): element.decompose()
            text = "\n".join(line.strip() for line in soup.get_text().splitlines() if line.strip())
            if not text or len(text) < 300:
                logging.warning(f"⚠️ (Requests) 未能從 {url} 提取到任何有效文字。")
                return "無法爬取網站，錯誤: 網頁內容過少或無效"
            logging.info(f"✅ (Requests) 成功獲取網頁文字，長度: {len(text)} 字元。")
            return text
        except Exception as e:
            logging.error(f"❌ (Requests) 失敗: {e}")
            return f"無法爬取網址，錯誤: {e}"

    def _perform_final_cleanup(self, task_id: str):
        if task_id:
            logging.info(f"🧹 任務 {task_id} 正在最終註銷。")
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

    def _send_done_signal(self) -> Generator[str, None, None]:
        yield f"data: {json.dumps({'type': 'done', 'content': '[DONE]'})}\n\n"
        logging.info("✅ 資料流已正常關閉。")

    def _handle_clarification(self, question: str, task_id: str) -> Generator[str, None, None]:
        logging.info(f"💬 KAIZEN 路由：問題過於寬泛，啟動問題澄清流程。")
        clarifier_template = self.prompts.get("query_clarifier")
        if not clarifier_template: raise ValueError("query_clarifier.txt 模板未找到！")
        clarifier_chain = clarifier_template | self.llm | StrOutputParser()
        clarification_str = clarifier_chain.invoke({"question": question})
        match = re.search(r"\{.*\}", clarification_str, re.DOTALL)
        if not match: raise ValueError(f"問題澄清師的回應中不包含有效的 JSON 結構: {clarification_str}")
        clarification_json = json.loads(match.group(0))
        yield f"data: {json.dumps({'type': 'clarification', 'data': clarification_json})}\n\n"

    def _handle_complex_project(self, question: str, stream: bool, task_id: str) -> Generator[str, None, None]:
        if not self.use_web_search:
            logging.info("🌐 網路研究功能已停用。轉向直接問答模式 (Invoke)。")
            direct_answer_template = self.prompts.get("direct_answer")
            direct_answer_prompt = direct_answer_template.format(question=question) if direct_answer_template else f"問題：{question}\n\n回答："
            logging.info("正在呼叫 LLM (invoke)...")
            full_response = self.llm.invoke(direct_answer_prompt)
            logging.info(f"LLM (invoke) 已回覆，內容長度: {len(str(full_response))}")
            response_content = full_response.content if hasattr(full_response, 'content') else str(full_response)
            if response_content and response_content.strip():
                yield f"data: {json.dumps({'type': 'content', 'content': response_content})}\n\n"
            else:
                logging.warning("⚠️ LLM 在直接問答模式下沒有生成任何內容。")
                fallback_message = "抱歉，我似乎對這個問題沒有想法。能否請您換個方式提問？"
                yield f"data: {json.dumps({'type': 'content', 'content': fallback_message})}\n\n"
            return

        logging.info(f"🚀 啟動 [KAIZEN 最終架構] 專家小組工作流 (Task ID: {task_id})...")
        def check_cancellation():
            if task_id and self.active_tasks.get(task_id, {}).get("is_cancelled"):
                logging.warning(f"🛑 任務 {task_id} 已被使用者取消。")
                raise InterruptedError(f"Task {task_id} cancelled by user.")
        check_cancellation()
        yield f"data: {json.dumps({'type': 'status', 'message': '步驟 1/4: 正在拆解研究任務...'})}\n\n"

        task_decomp_template = self.prompts.get("task_decomposition")
        if not task_decomp_template: raise ValueError("task_decomposition.txt 模板未找到！")
        task_decomp_prompt_string = task_decomp_template.format(question=question)
        sub_tasks_str = self.llm.invoke(task_decomp_prompt_string)
        matches = re.findall(r"^\s*\d+\.\s*(.*)", sub_tasks_str, re.MULTILINE)
        if not matches: raise ValueError("無法從 LLM 輸出中提取有效子任務。")
        validated_tasks = list(dict.fromkeys([t.strip() for t in matches]))[:4]

        if len(validated_tasks) < 2:
            logging.warning(f"任務拆解後有效任務不足2個，轉為直接回答模式。")
            direct_answer_prompt = self.prompts.get("direct_answer").format(question=question)
            full_response = self.llm.invoke(direct_answer_prompt)
            response_content = full_response.content if hasattr(full_response, 'content') else full_response
            yield f"data: {json.dumps({'type': 'content', 'content': response_content})}\n\n"
            return
        sub_tasks = validated_tasks
        logging.info(f"✅ 清理與驗證後的子任務 ({len(sub_tasks)} 條): {sub_tasks}")

        executive_summaries, all_source_documents = [], []
        analyst_template = self.prompts.get("research_synthesizer")
        summarizer_template = self.prompts.get("memo_summarizer")
        search_strategist_template = self.prompts.get("search_strategist")
        if not all([analyst_template, summarizer_template, search_strategist_template]): raise ValueError("一個或多個關鍵的 Prompt 模板未找到！")
        strategist_chain = search_strategist_template | self.llm | StrOutputParser()
        memo_summarizer_chain = summarizer_template | self.llm | StrOutputParser()

        for i, task in enumerate(sub_tasks):
            check_cancellation()
            yield f"data: {json.dumps({'type': 'status', 'message': f'步驟 2.{i+1}/{len(sub_tasks)}: 正在深度研究 \"{task[:20]}...\"'})}\n\n"
            search_queries_str = strategist_chain.invoke({"question": question, "task": task})
            search_queries = [line.strip() for line in re.findall(r"^\s*\d+\.\s*(.*)", search_queries_str, re.MULTILINE) if line.strip()] or [task]
            logging.info(f"   -> 策略師為 '{task}' 生成了 {len(search_queries)} 個搜尋向量: {search_queries}")
            task_specific_docs = []
            for query in search_queries:
                check_cancellation()
                docs_for_query = self._agent_based_web_search(query)
                if docs_for_query: task_specific_docs.extend(docs_for_query)
            context = "注意：未能從網路找到相關資料。"
            if task_specific_docs:
                all_source_documents.extend(task_specific_docs)
                unique_contents, unique_docs_for_synthesis = set(), []
                for doc in task_specific_docs:
                    if doc.page_content not in unique_contents:
                        unique_contents.add(doc.page_content)
                        unique_docs_for_synthesis.append(doc)
                logging.info(f"   -> 為子任務 '{task}' 匯總了 {len(unique_docs_for_synthesis)} 份不重複的文件進行綜合分析。")
                context = "\n---\n".join([f"來源：{doc.metadata.get('source', '網路')}\n內容：\n{doc.page_content}" for doc in unique_docs_for_synthesis])
            else:
                logging.warning(f"   -> 未能為子任務 '{task}' 找到任何網路資料。")
            detailed_memo = self.llm.invoke(analyst_template.format(context=context, question=task))
            check_cancellation()
            yield f"data: {json.dumps({'type': 'status', 'message': f'步驟 2.{i+1}/{len(sub_tasks)}: 正在精煉 \"{task[:20]}...\" 的研究成果...'})}\n\n"
            summary = memo_summarizer_chain.invoke({"memo": detailed_memo})
            executive_summaries.append(f"### 研究主題: {task}\n{summary}")
            logging.info(f"   -> 已為 '{task}' 生成執行摘要:\n{summary[:100]}...")
        final_context = "\n\n---\n\n".join(executive_summaries)
        
        check_cancellation()
        yield f"data: {json.dumps({'type': 'status', 'message': '步驟 3/4: 正在基於研究成果生成報告大綱...'})}\n\n"
        blueprint_gen_template = self.prompts.get("answer_blueprint_generator")
        if not blueprint_gen_template: raise ValueError("answer_blueprint_generator.txt 模板未找到！")
        base_blueprint_prompt = blueprint_gen_template.format(context=final_context, question=question)
        blueprint_json = None
        for attempt in range(3):
            check_cancellation()
            prompt_to_use = base_blueprint_prompt if attempt == 0 else f"{base_blueprint_prompt}\n\n[修正指令]: 上次解析失敗，請嚴格只輸出包裹在 ```json ``` 中的代碼塊，不要包含任何其他文字。"
            blueprint_str = self.llm.invoke(prompt_to_use)
            logging.info(f"--- 大綱生成器回應 (嘗試 {attempt + 1}) ---\n{blueprint_str}\n--------------------")
            try:
                match = re.search(r"```json\s*(\{.*?\})\s*```", blueprint_str, re.DOTALL) or re.search(r"(\{.*\})", blueprint_str, re.DOTALL)
                if not match: raise json.JSONDecodeError("輸出中找不到 JSON 結構。", blueprint_str, 0)
                blueprint_json = json.loads(match.group(1))
                logging.info("✅ 成功解析回答大綱 JSON。")
                break
            except json.JSONDecodeError as e:
                logging.warning(f"❌ 解析 JSON 失敗 (嘗試 {attempt + 1}): {e}")
        if blueprint_json is None: raise ValueError("在 3 次嘗試後仍無法解析 JSON。")
        
        check_cancellation()
        yield f"data: {json.dumps({'type': 'status', 'message': '步驟 4/4: 正在撰寫最終報告...'})}\n\n"
        final_writer_template = self.prompts.get("final_report_writer")
        if not final_writer_template: raise ValueError("關鍵的 'final_report_writer.txt' 模板未找到！")
        final_report_prompt = final_writer_template.format(question=question, context=final_context, blueprint=json.dumps(blueprint_json, indent=2, ensure_ascii=False))
        stream_iterator = self.llm.stream(final_report_prompt)
        for chunk in stream_iterator:
            check_cancellation()
            content_chunk = chunk.content if hasattr(chunk, 'content') else chunk
            yield f"data: {json.dumps({'type': 'content', 'content': content_chunk})}\n\n"
        logging.info("報告正文串流生成完畢！正在進行最終組裝...")
        unique_urls = {doc.metadata.get("source", "").replace("網頁瀏覽: ", "").strip() for doc in all_source_documents if doc.metadata.get("source", "").startswith("網頁瀏覽: ")}
        reference_list_str = "\n".join([f"- {url}" for url in sorted(list(unique_urls)) if url])
        if reference_list_str:
            reference_section_str = f"\n\n---\n\n## 參考文獻\n{reference_list_str}"
            logging.info(f"✅ 已提取 {len(unique_urls)} 條獨特的參考文獻。")
            yield f"data: {json.dumps({'type': 'content', 'content': reference_section_str})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'message': '報告生成完畢！'})}\n\n"

    def ask(self, question: str, stream: bool = True, task_id: str = None, bypass_assessment: bool = False) -> Generator[str, None, None]:
        if not task_id:
            task_id = f"task_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
        self.active_tasks[task_id] = {'is_cancelled': False}
        logging.info(f"\n🤔 收到請求並註冊任務 ID: {task_id}，問題: '{question}'")

        try:
            # --- 路由決策 ---
            if bypass_assessment:
                logging.info(f"🛂 KAIZEN 路由：收到繞過指令，直接啟動深度研究工作流。")
                yield from self._handle_complex_project(question, stream, task_id)

            elif not self.use_web_search:
                logging.info("🌐 KAIZEN 路由：網路研究已停用，強制執行直接問答。")
                yield from self._handle_complex_project(question, stream, task_id)

            elif re.search(r"https?://[\S]+", question):
                url = re.search(r"https?://[\S]+", question).group(0)
                logging.info(f"🔗 KAIZEN 路由：檢測到 URL，自動升格為深度研究任務。")
                rewritten_question = f"請為我撰寫一份關於以下網址內容的深度總結報告：{url}。報告需要提煉出其核心觀點、關鍵信息和主要論據。"
                yield from self._handle_complex_project(rewritten_question, stream, task_id)
            
            else:
                assessor_template = self.prompts.get("query_assessor")
                if not assessor_template: raise ValueError("query_assessor.txt 模板未找到！")
                assessor_chain = assessor_template | self.llm | StrOutputParser()
                assessment_str = assessor_chain.invoke({"question": question})
                match = re.search(r"\{.*\}", assessment_str, re.DOTALL)
                if not match: raise ValueError(f"問題評估師的回應中不包含有效的 JSON 結構: {assessment_str}")
                assessment_json = json.loads(match.group(0))
                assessment = assessment_json.get("assessment")
                logging.info(f"🧐 問題評估師結論: {assessment}")

                if assessment == "specific_topic":
                    yield from self._handle_complex_project(question, stream, task_id)
                elif assessment == "broad_concept":
                    yield from self._handle_clarification(question, task_id)
                else:
                    logging.warning(f"⚠️ 未知的評估結果: {assessment}。將直接執行深度研究。")
                    yield from self._handle_complex_project(question, stream, task_id)
            
            yield from self._send_done_signal()

        except GeneratorExit:
            logging.warning(f"🔌 客戶端在任務 {task_id} 執行期間斷開連接。正在靜默清理。")
        except InterruptedError:
            logging.info(f"🛑 任務 {task_id} 已被使用者成功中止。")
            yield f"data: {json.dumps({'type': 'status', 'message': '任務已取消。'})}\n\n"
            yield from self._send_done_signal()
        except Exception as e:
            logging.error(f"❌ 在 ASK 指揮官模式中發生嚴重錯誤: {e}", exc_info=True)
            error_message = f'抱歉，執行時發生了無法預期的錯誤: {str(e)}'
            yield f"data: {json.dumps({'type': 'error', 'content': error_message})}\n\n"
            yield from self._send_done_signal()
        finally:
            self._perform_final_cleanup(task_id)