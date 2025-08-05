import os
import json
import re
import requests
import wikipedia
from datetime import datetime
from bs4 import BeautifulSoup
import logging
import urllib3
import time
from typing import Any

# LangChain Imports
from ddgs import DDGS
from langchain_ollama.llms import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.utils import filter_complex_metadata

try:
    from playwright.sync_api import sync_playwright, Playwright, Browser
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning("Playwright 未安裝。爬蟲功能將僅使用 requests。建議執行 'pip install playwright && playwright install' 來增強爬蟲能力。")

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF 未安裝。PDF 處理功能將不可用。建議執行 'pip install PyMuPDF'。")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
        self.persist_directory = self.config["PERSIST_DIRECTORY"]
        self.use_web_search = True
        self.use_wikipedia = False
        self.use_history = False
        self.use_scraper = False
        self.ollama_base_url = self.config["OLLAMA_BASE_URL"]
        self.prompts = {}
        self.all_docs_for_bm25 = []
        self.bm25_retriever = None

        logging.info("正在初始化 Embedding 模型...")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config["EMBEDDING_MODEL_NAME"],
                model_kwargs={"device": self.config["EMBEDDING_DEVICE"], "trust_remote_code": True},
            )
        except Exception as e:
            logging.error(f"初始化 Embedding 模型失敗: {e}")
            raise

        logging.info("正在初始化/載入向量數據化...")
        try:
            self.vector_db = Chroma(
                persist_directory=self.persist_directory, embedding_function=self.embeddings
            )
        except Exception as e:
            logging.error(f"初始化向量數據庫失敗: {e}")
            raise

        logging.info("正在設置 LLM...")
        try:
            self.llm = OllamaLLM(model=self.config["llm_model"], base_url=self.ollama_base_url)
            self.current_llm_model = self.config["llm_model"]
        except Exception as e:
            logging.error(f"設置 LLM 失敗: {e}")
            raise
            
        self._init_prompts()

        logging.info("正在初始化混合檢索器...")
        try:
            self.vector_retriever = self.vector_db.as_retriever(search_kwargs={"k": self.config["VECTOR_SEARCH_K"]})
            self.ensemble_retriever = self.vector_retriever
            self.update_ensemble_retriever(full_rebuild=True)
        except Exception as e:
            logging.error(f"初始化混合檢索器失敗: {e}")
            raise

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
                    prompts[prompt_name] = PromptTemplate.from_template(f.read(), template_format="f-string")
                    logging.info(f"   -> 已加載: {prompt_name}")
        return prompts

    def _init_prompts(self):
        self.prompts = self._load_all_prompts("prompts")
        aux_prompts = {
            "query_expansion": "你是一個查詢優化助理。請根據使用者提供的原始查詢，生成一個更具體、更可能在技術檔中找到相關內容的擴展查詢。請只返回擴展後的查詢，不要添加任何解釋。\n\n[原始查詢]: {original_query}\n\n[擴展查詢]:",
            "web_search_generation": """你是一位精通網路搜尋的助理。你的任務是將使用者提出的「研究子任務」，結合「核心主題」，轉換成一組簡潔、高效的搜尋引擎關鍵字。

規則：
- 只返回關鍵字，不要包含任何解釋或多餘的文字。
- 始終包含核心主題，以確保搜尋結果的相關性。
- 使用繁體中文。

範例：
[核心主題]: 電動車市場分析
[研究子任務]: 分析電池技術的發展瓶頸
[關鍵字]: 電動車 電池技術瓶頸 固態電池 成本

[核心主題]: {question}  <-- 我們可以將原始問題作為核心主題
[研究子任務]: {task}
[關鍵字]:"""
        }
        for name, template_str in aux_prompts.items():
            if name not in self.prompts:
                self.prompts[name] = PromptTemplate.from_template(
                    template_str, template_format="f-string"
                )
                logging.info(f"   -> 已動態加載輔助模板: {name}")

        if 'router' not in self.prompts:
            logging.error("關鍵的 'router.txt' 模板未找到！請在 prompts 目錄中創建它。")
            fallback_router_template = '使用者問題: {question}\n\n[JSON]: {{"path": "rag_query", "persona": "default_rag"}}'
            self.prompts['router'] = PromptTemplate.from_template(
                fallback_router_template, template_format="f-string"
            )

    def set_llm_model(self, model_name: str):
        logging.info(f"\n🔄 正在切換 LLM 模型至: {model_name}")
        try:
            self.llm = OllamaLLM(model=model_name, base_url=self.ollama_base_url)
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
        if enabled:
            self.use_history = False
            self.use_wikipedia = False
            self.use_scraper = False
            logging.info("   -> (自動) 已停用歷史記錄、維基百科和URL爬蟲，進入純網路搜尋模式。")
        return True

    def set_history_retrieval(self, enabled: bool):
        logging.info(f"🔄 將歷史對話檢索設置為: {'啟用' if enabled else '停用'}")
        self.use_history = enabled
        if enabled:
            self.use_web_search = False
            logging.info("   -> (自動) 已停用網路搜尋。")
        return True

    def set_wikipedia_search(self, enabled: bool):
        logging.info(f"🔄 將維琪百科搜索設置為: {'啟用' if enabled else '停用'}")
        self.use_wikipedia = enabled
        if enabled:
            self.use_web_search = False
            logging.info("   -> (自動) 已停用網路搜尋。")
        return True

    def set_scraper_search(self, enabled: bool):
        logging.info(f"🔄 將網頁爬蟲設置為: {'啟用' if enabled else '停用'}")
        self.use_scraper = enabled
        if enabled:
            self.use_web_search = False
            logging.info("   -> (自動) 已停用網路搜尋。")
        return True

    def search_records(self, query: str = "", page: int = 1, per_page: int = 50):
        logging.info(f"🔍 正在資料庫中搜索 '{query}'，第 {page} 頁...")
        offset = (page - 1) * per_page
        where_document_filter = {"$contains": query} if query and query.strip() else None
        try:
            all_matching_ids = self.vector_db._collection.get(
                where_document=where_document_filter, include=[]
            )["ids"]
            total_records = len(all_matching_ids)
            results = self.vector_db.get(
                limit=per_page,
                offset=offset,
                where_document=where_document_filter,
                include=["metadatas", "documents"],
            )
            records = [
                {
                    "id": results["ids"][i],
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
                for i in range(len(results["ids"]))
            ]
            return {
                "records": sorted(records, key=lambda x: x["id"], reverse=True),
                "total_records": total_records,
                "current_page": page,
                "total_pages": (total_records + per_page - 1) // per_page,
            }
        except Exception as e:
            logging.error(f"在資料庫搜索時發生錯誤: {e}")
            return {
                "records": [],
                "total_records": 0,
                "current_page": 1,
                "total_pages": 0,
            }

    def update_ensemble_retriever(
        self, new_docs: list = None, full_rebuild: bool = False
    ):
        logging.info("🔄 正在更新 Ensemble Retriever...")
        if full_rebuild:
            logging.info("   -> 執行完整重建...")
            try:
                all_data = self.vector_db.get(include=["documents", "metadatas"])
                documents_content = all_data["documents"]
                metadatas = all_data["metadatas"]
                self.all_docs_for_bm25 = [
                    Document(page_content=documents_content[i], metadata=metadatas[i])
                    for i in range(len(documents_content))
                    if documents_content[i] != "start"
                ]
                logging.info(
                    f"   -> 從資料庫載入 {len(self.all_docs_for_bm25)} 份檔進行索引。"
                )
            except Exception as e:
                logging.error(f"從資料庫獲取文檔失敗: {e}")
                self.all_docs_for_bm25 = []
        if new_docs:
            logging.info(f"   -> 執行增量更新，新增 {len(new_docs)} 份文件...")
            self.all_docs_for_bm25.extend(new_docs)
        if not self.all_docs_for_bm25:
            logging.info("   -> 資料庫文檔不足，僅使用向量檢索器。")
            self.ensemble_retriever = self.vector_retriever
            self.bm25_retriever = None
            return
        try:
            logging.info(
                f"   -> 正在基於 {len(self.all_docs_for_bm25)} 份檔重建 BM25 索引..."
            )
            self.bm25_retriever = BM25Retriever.from_documents(
                self.all_docs_for_bm25, k=self.config["BM25_SEARCH_K"]
            )
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_retriever],
                weights=self.config["ENSEMBLE_WEIGHTS"],
            )
            logging.info("✅ 混合檢索器已成功更新。")
        except Exception as e:
            logging.error(f"更新混合檢索器失敗: {e}。將退回至僅使用向量檢索器。")
            self.ensemble_retriever = self.vector_retriever

    def add_document(self, file_path: str):
        logging.info(f"📄 正在處理新文件: {file_path}")
        try:
            loader = UnstructuredLoader(file_path, mode="elements", strategy="fast")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.get("CHUNK_SIZE", 1000),
                chunk_overlap=self.config.get("CHUNK_OVERLAP", 200),
            )
            splits = text_splitter.split_documents(docs)
            logging.info(f"   -> 文件被切割成 {len(splits)} 個片段。")
            file_name = os.path.basename(file_path)
            for split in splits:
                split.metadata["source"] = file_name
            final_splits = filter_complex_metadata(splits)
            logging.info(f"   -> 已清理 {len(final_splits)} 個片段的元數據。")
            if not final_splits:
                logging.warning("   -> 文件處理後沒有生成任何可用的文字片段，處理中止。")
                return
            batch_size = 32
            total_splits = len(final_splits)
            logging.info(f"   -> 將以每批 {batch_size} 個片段的大小，分批次存入資料庫...")
            for i in range(0, total_splits, batch_size):
                batch = final_splits[i : i + batch_size]
                self.vector_db.add_documents(batch)
                logging.info(
                    f"      -> 已存入 {min(i + batch_size, total_splits)} / {total_splits} 個片段..."
                )
            logging.info(f"✅ 檔 '{file_name}' 已成功存入向量資料庫。")
            self.update_ensemble_retriever(new_docs=final_splits)
        except Exception as e:
            logging.error(f"文件處理時發生嚴重錯誤: {e}", exc_info=True)
            raise
        finally:
            logging.info(f"🧹 正在清理暫存檔: {file_path}")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    logging.error(f"清理暫存檔 {file_path} 失敗: {e}")

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
                    # 獲取更多結果以便過濾
                    search_results = ddgs.text(question, max_results=20, region="tw-zh")
                if not search_results: return []
                
                # --- [核心修改] 過濾掉黑名單中的網址 ---
                urls_to_browse = [
                    res["href"] for res in search_results 
                    if "href" in res and not any(blacklisted in res["href"] for blacklisted in DOMAIN_BLACKLIST)
                ][:10] # 取過濾後的前10個
                # ------------------------------------------

                logging.info(f"   -> 已過濾掉黑名單網站，準備瀏覽以下 {len(urls_to_browse)} 個網頁...")
                if not urls_to_browse:
                    logging.warning("   -> 過濾後沒有可瀏覽的網頁。")
                    return []

                for url in urls_to_browse:
                    if not url: continue
                    content = self._scrape_webpage_text(url, browser)
                    if "無法爬取" not in content and len(content) > 100: # 新增長度判斷，過濾掉無用內容
                        all_source_docs.append(Document(page_content=content, metadata={"source": f"網頁瀏覽: {url}"}))
                
                return all_source_docs
            except Exception as e:
                logging.error(f"❌ 在 Agent 搜尋過程中發生錯誤: {e}", exc_info=True)
                return []
            finally:
                if browser: browser.close()
                logging.info("Agent 搜尋任務完成，Playwright 資源已釋放。")

    def _scrape_webpage_text(self, url: str, browser: Browser = None):
        if url.lower().endswith('.pdf'):
            if not PYMUPDF_AVAILABLE:
                return "無法處理 PDF 文件，因為 PyMuPDF 未安裝。"
            try:
                logging.info(f"📄 (PyMuPDF模式) 正在處理 PDF 網址: {url}")
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'}
                response = requests.get(url, headers=headers, timeout=30, verify=False)
                response.raise_for_status()
                
                pdf_document = fitz.open(stream=response.content, filetype="pdf")
                text = "".join(page.get_text() for page in pdf_document)
                pdf_document.close()
                logging.info(f"✅ (PyMuPDF) 成功提取 PDF 文字，長度: {len(text)} 字元。")
                return text
            except Exception as e:
                logging.error(f"❌ (PyMuPDF) 處理 PDF 時失敗: {e}")
                return f"無法爬取 PDF，錯誤: {e}"

        if browser:
            logging.info(f"🕸️ (Playwright模式) 正在嘗試爬取網址: {url}")
            page = None
            try:
                if any(url.lower().endswith(ext) for ext in ['.doc', '.docx', '.zip', '.rar', '.xls', '.xlsx']):
                     raise Exception(f"文件類型 ({url})，跳過 Playwright。")
                page = browser.new_page()
                page.goto(url, timeout=30000, wait_until='domcontentloaded')

                try:
                    page.wait_for_selector(
                        "main, article, #content, #main-content, .post-content, .article-body", 
                        timeout=10000
                    )
                    logging.info("   -> ✅ 成功等到關鍵內容區塊。")
                except Exception as wait_e:
                    logging.warning(f"   -> ⚠️ 未能等到特定內容區塊，將直接抓取現有內容。錯誤: {wait_e}")
                html_content = page.content()
                soup = BeautifulSoup(html_content, "html.parser")
                for element in soup(["script", "style", "nav", "footer", "aside", "header", "iframe", "form"]):
                    element.decompose()
                text = "\n".join(line.strip() for line in soup.get_text().splitlines() if line.strip())
                logging.info(f"✅ (Playwright) 成功獲取網頁文字，長度: {len(text)} 字元。")
                return text
            except Exception as e:
                logging.warning(f"❌ (Playwright) 失敗: {e}。將回退至 Requests 模式。")
            finally:
                if page and not page.is_closed():
                    page.close()
        
        logging.info(f"🕸️ (Requests模式) 正在嘗試爬取網址: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=30, verify=False)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "aside", "header", "iframe", "form"]):
                element.decompose()
            text = "\n".join(line.strip() for line in soup.get_text().splitlines() if line.strip())
            if not text:
                logging.warning(f"⚠️ (Requests) 未能從 {url} 提取到任何有效文字。")
            else:
                logging.info(f"✅ (Requests) 成功獲取網頁文字，長度: {len(text)} 字元。")
            return text
        except Exception as e:
            logging.error(f"❌ (Requests) 失敗: {e}")
            return f"無法爬取網址，錯誤: {e}"

    def _search_wikipedia(self, query: str):
        logging.info(f"🔍 (維琪百科) 正在搜索: '{query[:50].strip()}...'")
        try:
            wikipedia.set_lang("zh-tw")
            summary = wikipedia.summary(query, sentences=5, auto_suggest=False)
            return summary
        except Exception:
            return "無相關資料"

    def _get_rag_context(self, question: str, retrieval_query: str):
        all_source_docs, context_parts = [], []
        if self.use_scraper:
            if url_match := re.search(r"https?://[\S]+", question):
                url = url_match.group(0)
                web_content = self._scrape_webpage_text(url, browser=None)
                if "無法爬取" not in web_content:
                    doc = Document(page_content=web_content, metadata={"source": f"網頁: {url}"})
                    all_source_docs.append(doc)
                    context_parts.append(f"來源：{doc.metadata['source']}\n內容：\n{doc.page_content}")
        if self.use_wikipedia:
            wiki_content = self._search_wikipedia(question)
            if "無相關資料" not in wiki_content:
                doc = Document(page_content=wiki_content, metadata={"source": "維琪百科"})
                all_source_docs.append(doc)
                context_parts.append(f"來源：{doc.metadata['source']}\n內容：\n{doc.page_content}")
        if self.use_history:
            db_docs = self.ensemble_retriever.get_relevant_documents(retrieval_query)
            db_docs = [doc for doc in db_docs if doc.page_content != "start"]
            if db_docs:
                all_source_docs.extend(db_docs)
                context_parts.append("[相關資料庫內容]:\n" + "\n---\n".join([f"來源：{doc.metadata.get('source', '未知')}\n內容：\n{doc.page_content}" for doc in db_docs]))
        return "\n\n".join(context_parts) if context_parts else "沒有可用的上下文資料。", all_source_docs

    def ask(self, question: str, stream: bool = True):
        logging.info(f"\n🤔 收到請求，問題: '{question}'")
        
        # --- 步驟 1: 路由器決策 ---
        router_template = self.prompts.get('router')
        router_prompt_string = router_template.format(question=question)
        path, persona = "rag_query", "default_rag" # 預設值
        try:
            raw_route_output = self.llm.invoke(router_prompt_string)
            logging.info(f"🚦 路由器原始決策: {raw_route_output}")
            match = re.search(r'{\s*"path"\s*:\s*".*?",\s*"persona"\s*:\s*".*?"\s*}', raw_route_output, re.DOTALL)
            if match:
                decision = json.loads(match.group(0))
                path, persona = decision.get("path", path), decision.get("persona", persona)
            else:
                logging.warning("無法從路由器輸出解析 JSON，退回預設路徑。")
            logging.info(f"🚦 路由器清理後決策 -> 路徑: '{path}', 角色: '{persona}'")
        except Exception as e:
            logging.error(f"🚦 路由器決策失敗: {e}, 將走預設 RAG 路徑。")

        # --- 步驟 2: 根據路由執行對應工作流 (嚴格的 if/elif/else 結構) ---
        main_prompt_template = self.prompts.get(persona) or self.prompts.get('default_rag')
        if not main_prompt_template:
            return self._stream_direct_message("系統錯誤：找不到對應的角色模板。")

        if path == "complex_project_query":
            # 複雜研究：只執行專家小組工作流，該流程內部只會進行即時網路研究，
            # 完美實現了「專注模式」。
            return self._handle_complex_project(question, stream)

        elif path == "web_search_query":
            # 通用網路搜尋：只在使用者啟用時執行。
            logging.info(f"🌐 走通用網路搜索 RAG 路徑 (使用角色: {persona})...")
            if not self.use_web_search:
                 return self._stream_direct_message("網路搜尋功能目前已停用。")
                 
            keyword_chain = self.prompts['web_search_generation'] | self.llm | StrOutputParser()
            # 注意：對於簡單Web搜尋，task和question是同一個
            search_keywords_raw = keyword_chain.invoke({"question": question, "task": question})
            search_keywords = re.sub(r"<think>.*?</think>|\[.*?\]:", "", search_keywords_raw, flags=re.DOTALL).strip()
            
            search_docs = self._agent_based_web_search(search_keywords)
            if not search_docs:
                return self._stream_direct_message("抱歉，我嘗試透過網路搜尋，但未能獲取到相關資訊。")
                
            context = "[網路搜索結果]:\n" + "\n---\n".join([f"來源：{doc.metadata.get('source', '網路')}\n內容：\n{doc.page_content}" for doc in search_docs])
            prompt = main_prompt_template.format(context=context, question=question)
            return self.stream_and_save(question, prompt, search_docs)

        elif path == "rag_query":
            # 本地知識庫查詢：只查詢本地資料庫，混合歷史、維基等。
            logging.info(f"📚 走本地 RAG 路徑 (使用角色: {persona})...")
            query_chain = self.prompts['query_expansion'] | self.llm | StrOutputParser()
            expanded_query = query_chain.invoke({"original_query": question}).strip()
            retrieval_query = f"{question}\n{expanded_query}"
            
            # _get_rag_context 會根據 use_history, use_wikipedia 等開關決定上下文
            context, source_docs = self._get_rag_context(question, retrieval_query)
            prompt = main_prompt_template.format(context=context, question=question)
            return self.stream_and_save(question, prompt, source_docs)

        else: # direct_answer_query or general_conversation
            # 直接回答或通用對話：不使用任何外部資料。
            logging.info(f"💬 走直接/通用對話路徑 (使用角色: {persona})...")
            prompt = main_prompt_template.format(context="沒有可用的上下文資料。", question=question)
            return self.stream_and_save(question, prompt, [])

    def _write_report_from_blueprint(self, blueprint_json: dict, full_context: str, question: str):
        """
        [V7 - 終極自適應解析器版本]
        基於原則進行解析，能夠處理多種不斷進化的 JSON 結構。
        核心原則：利用 AI 生成的「目錄」或「章節列表」作為尋找章節的路線圖。
        """
        logging.info("➡️ 進入 [V7-終極自適應] 章節撰寫器工作流...")
        
        chapter_writer_template = self.prompts.get('final_chapter_writer')
        if not chapter_writer_template:
            # 確保你已經創建並更新了 final_chapter_writer.txt
            chapter_writer_template = self.prompts.get('chapter_writer') 
            logging.warning("未找到 'final_chapter_writer.txt'，退回使用 'chapter_writer.txt'。")

        # --- [終極自適應解析邏輯] ---
        chapters_to_write = []

        # 策略 1: 尋找名為 '目錄' 或 'table_of_contents' 的列表，並用它作為地圖
        toc_key = next((key for key in ['目录', 'table_of_contents', 'contents'] if key in blueprint_json and isinstance(blueprint_json[key], list)), None)
        if toc_key:
            logging.info(f"   -> 解析策略 1: 成功匹配 '{toc_key}' 列表結構，將其用作路線圖。")
            for chapter_title in blueprint_json[toc_key]:
                # 從 JSON 的頂層尋找與目錄標題完全匹配的鍵
                if chapter_title in blueprint_json and isinstance(blueprint_json[chapter_title], dict):
                    chapters_to_write.append((chapter_title, blueprint_json[chapter_title]))
        
        # 策略 2: 如果沒有目錄，尋找名為 'sections' 的列表
        elif 'sections' in blueprint_json and isinstance(blueprint_json['sections'], list):
            logging.info("   -> 解析策略 2: 成功匹配 'sections' 列表結構。")
            for item in blueprint_json['sections']:
                if isinstance(item, dict) and 'section_title' in item:
                    chapters_to_write.append((item['section_title'], item.get('subsections', {})))

        # 策略 3: 作為最終的降級方案，收集所有值為字典的頂層鍵
        else:
            logging.warning("   -> 未找到明確的目錄或章節列表，啟用降級策略：掃描所有頂層字典。")
            chapters_to_write = [(key, value) for key, value in blueprint_json.items() if isinstance(value, dict)]

        if not chapters_to_write:
            logging.error("大綱 JSON 結構無法識別，或解析後章節列表為空。")
            yield "## 大綱錯誤\n\n報告生成失敗，因為生成的大綱結構無法識別或為空。"
            return
        # ----------------------------------------------------

        is_first_chapter = True
        for chapter_title, chapter_data in chapters_to_write:
            try:
                if not is_first_chapter:
                    yield "\n\n---\n\n"
                is_first_chapter = False

                key_points_str = self._generate_markdown_from_blueprint(chapter_data, level=1)

                logging.info(f"   -> 正在為章節 '{chapter_title}' 撰寫內容並串流輸出...")
                
                chapter_prompt = chapter_writer_template.format(
                   question=question,
                   context=full_context,
                   chapter_title=chapter_title,
                   key_points=key_points_str
                )
                
                raw_chapter_content = ""
                for chunk in self.llm.stream(chapter_prompt):
                    # 在這裡進行最終的清理，以防萬一
                    cleaned_chunk = re.sub(r"<think>.*?</think>", "", chunk, flags=re.DOTALL).strip(" \n")
                    if cleaned_chunk:
                        # 確保 chunk 之間有空格，避免單詞粘連
                        raw_chapter_content += cleaned_chunk + "" 
                        yield cleaned_chunk

                if not raw_chapter_content.strip():
                    logging.warning(f"   -> ⚠️ 章節 '{chapter_title}' 的生成內容為空，已跳過。")
                    yield f"## {chapter_title}\n\n_{{此章節生成內容為空}}_\n\n"

            except Exception as e:
                logging.error(f"❌ 在撰寫章節 '{chapter_title}' 時發生錯誤: {e}", exc_info=True)
                yield f"## {chapter_title}\n\n_{{此章節生成失敗: {e}}}_\n\n"
        
        logging.info("✅ 所有章節已串流輸出完畢。")

    def _generate_markdown_from_blueprint(self, node: Any, level: int = 0) -> str:
        """
        遞迴地將大綱的 JSON 節點轉換為格式化的 Markdown 字串。
        """
        parts = []
        indent = "  " * level  # 根據層級縮排

        if isinstance(node, dict):
            for key, value in node.items():
                # 將 key 作為一個小標題或重點
                parts.append(f"{indent}- **{key}:**")
                # 遞迴處理 value，層級+1
                parts.append(self._generate_markdown_from_blueprint(value, level + 1))
        elif isinstance(node, list):
            for item in node:
                # 列表中的每個項目都遞迴處理
                parts.append(self._generate_markdown_from_blueprint(item, level))
        else:
            # 如果是字串或數字等基本類型，直接作為列表項
            parts.append(f"{indent}- {str(node)}")
            
        return "\n".join(parts)

    def _handle_complex_project(self, question: str, stream: bool = True):
      try:
          logging.info("🚀 啟動 [V8-最終架構] 專家小組工作流...")
          yield f"data: {json.dumps({'type': 'status', 'message': '步驟 1/3: 正在拆解與研究任務...'})}\n\n"
        
          # --- 步驟 1 & 2: 任務拆解與研究 (保持不變) ---
          task_decomp_template = self.prompts.get('task_decomposition')
          task_decomp_prompt_string = task_decomp_template.format(question=question)
          sub_tasks_str = self.llm.invoke(task_decomp_prompt_string)
          matches = re.findall(r"^\s*\d+\.\s*(.*)", sub_tasks_str, re.MULTILINE)
          if not matches: raise ValueError("無法從 LLM 輸出中提取有效子任務。")
          unique_tasks = list(dict.fromkeys([t.strip() for t in matches]))
          validated_tasks = unique_tasks[:4]
          if len(validated_tasks) < 2: raise ValueError(f"任務拆解後有效任務不足2個。")
          sub_tasks = validated_tasks
          logging.info(f"✅ 清理與驗證後的子任務 ({len(sub_tasks)} 條): {sub_tasks}")

          research_memos = []
          all_source_documents = []
          analyst_template = self.prompts.get('research_synthesizer')
          keyword_gen_chain = self.prompts['web_search_generation'] | self.llm | StrOutputParser()

          for i, task in enumerate(sub_tasks):
              yield f"data: {json.dumps({'type': 'status', 'message': f'步驟 1.{i+1}/{len(sub_tasks)}: 正在研究 \"{task[:20]}...\"'})}\n\n"
              search_keywords = keyword_gen_chain.invoke({"question": question, "task": task}).strip()
              search_docs = self._agent_based_web_search(search_keywords)
              if search_docs: all_source_documents.extend(search_docs)
              context = "\n---\n".join([f"來源：{doc.metadata.get('source', '網路')}\n內容：\n{doc.page_content}" for doc in search_docs]) if search_docs else "注意：未能從網路找到相關資料。"
              memo = self.llm.invoke(analyst_template.format(context=context, question=task))
              research_memos.append(f"--- 研究備忘錄 for '{task}' ---\n{memo}\n")
          final_context = "\n".join(research_memos)

          # --- 步驟 3: 生成最終大綱 (保持不變) ---
          yield f"data: {json.dumps({'type': 'status', 'message': '步驟 2/3: 正在生成最終報告大綱...'})}\n\n"
          format_example_str = "..." # (保持你現有的通用範例字串即可)
          blueprint_gen_template = self.prompts.get('answer_blueprint_generator')
          base_blueprint_prompt = blueprint_gen_template.format(format_example=format_example_str, context=final_context, question=question)
          blueprint_json = None
          # ... (保持現有的 JSON 解析與重試邏輯)
          for attempt in range(3):
              prompt_to_use = base_blueprint_prompt if attempt == 0 else f"{base_blueprint_prompt}\n\n[修正指令]: 上次解析失敗，請嚴格輸出 JSON。"
              blueprint_str = self.llm.invoke(prompt_to_use)
              try:
                  match = re.search(r"```json\s*(\{.*?\})\s*```", blueprint_str, re.DOTALL) or re.search(r"(\{.*\})", blueprint_str, re.DOTALL)
                  if not match: raise json.JSONDecodeError("輸出中找不到 JSON。", blueprint_str, 0)
                  blueprint_json = json.loads(match.group(1))
                  logging.info(f"✅ 成功解析回答大綱 JSON。")
                  break
              except json.JSONDecodeError as e:
                  logging.warning(f"❌ 解析 JSON 失敗 (嘗試 {attempt + 1}): {e}")
          if blueprint_json is None: raise ValueError("在 3 次嘗試後仍無法解析 JSON。")

          # --- 步驟 4: [最終架構] - 單一的、權威的總報告生成器 ---
          yield f"data: {json.dumps({'type': 'status', 'message': '步驟 3/3: 正在撰寫最終報告...'})}\n\n"
          
          final_writer_template = self.prompts.get('final_chapter_writer')
          final_report_prompt = final_writer_template.format(
              context=final_context,
              blueprint=json.dumps(blueprint_json, indent=2, ensure_ascii=False) # 將漂亮的 JSON 大綱作為字串傳入
          )

          full_answer = ""
          # 直接串流原始輸出，不做任何中間處理
          for chunk in self.llm.stream(final_report_prompt):
              full_answer += chunk
              yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
          
          logging.info("報告串流生成完畢！正在進行最終清理與儲存...")

          # 在儲存前，對完整的答案進行一次性的、徹底的清理
          final_cleaned_answer = re.sub(r"<think>.*?</think>", "", full_answer, flags=re.DOTALL).strip()
          
          if all_source_documents:
              yield f"data: {json.dumps({'type': 'sources', 'data': [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in all_source_documents]})}\n\n"

          if final_cleaned_answer:
              self.save_qa(question, final_cleaned_answer)
          
          yield f"data: {json.dumps({'type': 'status', 'message': '報告生成完畢！'})}\n\n"

      except Exception as e:
          logging.error(f"❌ 在專家小組工作流中發生嚴重錯誤: {e}", exc_info=True)
          yield f"data: {json.dumps({'type': 'error', 'error': f'抱歉，執行複雜專案時發生錯誤: {e}'})}\n\n"
      finally:
          yield f"data: [DONE]\n\n"

    def save_qa(self, question, answer):
        if not answer or not answer.strip():
            return
        qa_pair_content = f"問題: {question}\n回答: {answer}"
        metadata = {"source": "conversation", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        new_doc = Document(page_content=qa_pair_content, metadata=metadata)
        self.vector_db.add_documents([new_doc])
        self.update_ensemble_retriever(new_docs=[new_doc])
        logging.info("   -> 對話歷史存儲並同步至混合索引完畢！")
