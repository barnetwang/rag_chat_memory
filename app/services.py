import os
import json
import re
import requests
import wikipedia
from datetime import datetime
from bs4 import BeautifulSoup

# LangChain Imports
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

def get_ollama_models(ollama_base_url="http://localhost:11434"):
    try:
        response = requests.get(f"{ollama_base_url}/api/tags")
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]
    except Exception as e:
        print(f"❌ 獲取 Ollama 模型時發生錯誤: {e}")
        return []

class ConversationalRAG:
    def __init__(self, config: dict):
        self.config = config
        self.persist_directory = self.config['PERSIST_DIRECTORY']
        self.use_wikipedia = True
        self.use_history = True
        self.use_scraper = True
        self.ollama_base_url = self.config['OLLAMA_BASE_URL']
        self.history_summary_threshold = 2000

        # CORRECTED: Use spaces for indentation
        self.all_docs_for_bm25 = []
        self.bm25_retriever = None

        print("正在初始化 Embedding 模型...")
        print(f"   -> 使用模型: {self.config['EMBEDDING_MODEL_NAME']}")
        print(f"   -> 使用設備: {self.config['EMBEDDING_DEVICE']}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['EMBEDDING_MODEL_NAME'], 
            model_kwargs={'device': self.config['EMBEDDING_DEVICE'], 'trust_remote_code': True}
        )
        print("正在初始化/載入向量資料庫...")
        self.vector_db = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embeddings
        )
        print("正在設定 LLM...")
        self.llm = OllamaLLM(model=self.config['llm_model'], base_url=self.ollama_base_url)
        self.current_llm_model = self.config['llm_model']

        # --- Hybrid retriever initialization ---
        print("🔧 正在初始化混合檢索器...")
        self.vector_retriever = self.vector_db.as_retriever(
            search_kwargs={'k': self.config['VECTOR_SEARCH_K']}
        )
        self.ensemble_retriever = self.vector_retriever # Start with vector retriever only
        # The full ensemble retriever will be built by update_ensemble_retriever
        self.update_ensemble_retriever(full_rebuild=True)
        
        # --- Prompt Templates 初始化 ---
        self._init_prompts()
    
    def _init_prompts(self):
        self.main_prompt = PromptTemplate.from_template(
            """你是一位頂級的 AI 技術文件分析師。你的任務是嚴格基於下面提供的「上下文資料」，深入地回答「使用者當前問題」。

**執行流程與規則：**
1.  **深入分析上下文**：首先，仔細閱讀並完全理解「上下文資料」中與「使用者當前問題」相關的所有片段。
2.  **组织与回答**：
    *   如果「上下文資料」中包含回答問題所需的資訊，你的任務是【授權並鼓勵】你使用自己的語言能力，對這些碎片化的資訊進行**總結、推理、和重新組織**，以形成一個連貫、清晰、專業的回答。
    *   如果「上下文資料」完全沒有提及問題的核心主題，那麼你【必須】只回答：「根據我所掌握的資料，我找不到關於 '{question}' 的確切資訊。」
3.  **定义“幻觉”禁区**：你【絕對禁止】引入任何**在「上下文資料」中完全不存在的、憑空捏造的事實或數據**。
4.  **思考過程**：在最终答案前，你可以使用 <think>...</think> 標籤来写下你的分析、推理和判断过程。
5.  **【格式要求】**: `<think>` 區塊必須以 `</think>` 結束，且 `</think>` 標籤之後【必須立刻】开始你的最终回答，中间不能有任何换行或多余的空格。

---
[上下文資料]:
{context}
---
[使用者當前問題]: {question}
你的回答:"""
        )
        self.query_expansion_prompt = PromptTemplate.from_template(
            "你是一個查詢優化助理。請根據使用者提供的原始查詢，生成一個更具體、更可能在技術文件中找到相關內容的擴充查詢。請只返回擴充後的查詢，不要添加任何解釋。\n\n[原始查詢]: {original_query}\n\n[擴充查詢]:"
        )
        self.router_prompt = PromptTemplate.from_template(
            """你是一個任務路由器。根據使用者的問題，判斷它屬於哪一種類型。請只回答以下分類中的一個：'rag_query' 或 'general_conversation'。

- 如果問題需要查找、解釋、比較或定義特定資訊，特別是技術術語，請回答 'rag_query'。
- 如果問題是簡單的問候、閒聊或與知識庫無關的對話，請回答 'general_conversation'。

[使用者問題]: {question}
[分類]:"""
        )

    def set_llm_model(self, model_name: str):
        print(f"\n🔄 正在切換 LLM 模型至: {model_name}")
        try:
            self.llm = OllamaLLM(model=model_name, base_url=self.ollama_base_url)
            self.llm.invoke("Hi", stop=["Hi"]) 
            self.current_llm_model = model_name
            print(f"✅ LLM 模型成功切換為: {self.current_llm_model}")
            return True
        except Exception as e:
            print(f"❌ 切換 LLM 模型失敗: {e}")
            return False

    def set_history_retrieval(self, enabled: bool):
        print(f"🔄 將歷史對話檢索設定為: {'啟用' if enabled else '停用'}")
        self.use_history = enabled
        return True

    def set_wikipedia_search(self, enabled: bool):
        print(f"🔄 將維基百科搜尋設定為: {'啟用' if enabled else '停用'}")
        self.use_wikipedia = enabled
        return True
    
    def set_scraper_search(self, enabled: bool):
        print(f"🔄 將網頁爬蟲設定為: {'啟用' if enabled else '停用'}")
        self.use_scraper = enabled
        return True

    # --- MAJOR REFACTORING of update_ensemble_retriever ---
    def update_ensemble_retriever(self, new_docs: list = None, full_rebuild: bool = False):
        print("🔄 正在更新 Ensemble Retriever...")
        if full_rebuild:
            print("   -> 執行完整重建...")
            try:
                all_data = self.vector_db.get(include=["documents", "metadatas"])
                documents_content = all_data['documents']
                metadatas = all_data['metadatas']
                self.all_docs_for_bm25 = [
                    Document(page_content=documents_content[i], metadata=metadatas[i])
                    for i in range(len(documents_content))
                    if documents_content[i] != "start"
                ]
                print(f"   -> 從資料庫載入 {len(self.all_docs_for_bm25)} 份文件進行索引。")
            except Exception as e:
                print(f"❌ 從 DB 獲取文檔失敗: {e}")
                self.all_docs_for_bm25 = []

        if new_docs:
            print(f"   -> 執行增量更新，新增 {len(new_docs)} 份文件...")
            self.all_docs_for_bm25.extend(new_docs)

        if not self.all_docs_for_bm25:
            print("   -> 資料庫文檔不足，僅使用向量檢索器。")
            self.ensemble_retriever = self.vector_retriever
            self.bm25_retriever = None
            return

        try:
            print(f"   -> 正在基於 {len(self.all_docs_for_bm25)} 份文件重建 BM25 索引...")
            self.bm25_retriever = BM25Retriever.from_documents(
                self.all_docs_for_bm25,
                k=self.config['BM25_SEARCH_K']
            )
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_retriever],
                weights=self.config['ENSEMBLE_WEIGHTS']
            )
            print("✅ 混合檢索器已成功更新。")
        except Exception as e:
            print(f"❌ 更新混合檢索器失敗: {e}。將退回至僅使用向量檢索器。")
            self.ensemble_retriever = self.vector_retriever
            
    def add_document(self, file_path: str):
        print(f"📄 正在處理新文件: {file_path}")
        try:
            loader = UnstructuredLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['CHUNK_SIZE'], 
                chunk_overlap=self.config['CHUNK_OVERLAP']
            )
            splits = text_splitter.split_documents(docs)
            self.vector_db.add_documents(splits)
            print(f"✅ 文件 '{os.path.basename(file_path)}' 的分塊已成功加入向量資料庫。")
            self.update_ensemble_retriever(new_docs=splits)
        finally:
            # --- 無論 try 區塊成功或失敗，這裡都保證會執行 ---
            print(f"🧹 正在清理暫存文件: {file_path}")
            if os.path.exists(file_path):
                os.remove(file_path)
            
    def _scrape_webpage_text(self, url: str):
        print(f"🕸️ (爬蟲) 正在嘗試爬取網址: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'aside']): element.decompose()
            text = '\n'.join(line.strip() for line in soup.get_text().splitlines() if line.strip())
            print(f"✅ (爬蟲) 成功獲取網頁文字，長度: {len(text)} 字元。")
            return text[:4000]
        except Exception as e:
            print(f"❌ (爬蟲) 失敗: {e}")
            return f"無法爬取網址，錯誤: {e}"
            
    def _search_wikipedia(self, query: str):
        print(f"🔍 (維基百科) 正在搜尋: '{query[:50].strip()}...'")
        try:
            wikipedia.set_lang("zh-tw")

            summary = wikipedia.summary(query, sentences=5, auto_suggest=False)
            print(f"✅ (維基百科) 找到摘要。")
            return summary
        except wikipedia.exceptions.PageError:
            return "無相關資料"
        except Exception as e:
            print(f"❌ (維基百科) 查詢時發生錯誤: {e}")
            return "查詢時發生錯誤"

    def _get_rag_context(self, question: str, retrieval_query: str):
        all_source_docs = []
        context_parts = []

        # 1. 網頁爬蟲
        if self.use_scraper:
            url_match = re.search(r'https?://[\S]+', question)
            if url_match:
                url = url_match.group(0)
                web_content = self._scrape_webpage_text(url)
                if "無法爬取" not in web_content:
                    web_doc = Document(page_content=web_content, metadata={"source": f"網頁: {url}"})
                    all_source_docs.append(web_doc)
                    context_parts.append(f"來源：{web_doc.metadata['source']}\n內容：\n{web_doc.page_content}")
        if self.use_wikipedia:
            short_query = question
            wiki_content = self._search_wikipedia(short_query)
            if wiki_content not in ["無相關資料", "查詢時發生錯誤"]:
                wiki_doc = Document(page_content=wiki_content, metadata={"source": "維基百科"})
                all_source_docs.append(wiki_doc)
                context_parts.append(f"來源：{wiki_doc.metadata['source']}\n內容：\n{wiki_doc.page_content}")

        if self.use_history:
            print(f"🔍 (混合檢索) 正在檢索: '{retrieval_query[:80].replace('\n', ' ')}...'")
            db_docs = self.ensemble_retriever.get_relevant_documents(retrieval_query)
            db_docs = [doc for doc in db_docs if hasattr(doc, 'page_content') and doc.page_content != "start"]
            
            if db_docs:
                all_source_docs.extend(db_docs)
                context_from_docs = "\n---\n".join([f"來源：{doc.metadata.get('source', '未知')}\n內容：\n{doc.page_content}" for doc in db_docs])
                context_parts.append(f"[相關資料庫內容]:\n{context_from_docs}")
        
        final_context = "\n\n".join(context_parts) if context_parts else "沒有可用的上下文資料。"
        return final_context, all_source_docs

    def ask(self, question: str, stream: bool = True):
        print(f"\n🤔 收到請求，問題: '{question}'")

        router_chain = self.router_prompt | self.llm | StrOutputParser()
        try:
            route = router_chain.invoke({"question": question}).strip().lower()
            print(f"🚦 路由器決策: {route}")
        except Exception as e:
            print(f"🚦 路由器決策失敗: {e}, 將走預設 RAG 路徑。")
            route = 'rag_query'

        if 'rag_query' in route:
            try:
                query_chain = self.query_expansion_prompt | self.llm | StrOutputParser()
                raw_expanded_output = query_chain.invoke({"original_query": question})
                match = re.search(r'\[擴充查詢\]:\s*([\s\S]*)', raw_expanded_output, re.IGNORECASE)
                if match:
                    expanded_query = match.group(1).strip()
                else:
                    expanded_query = re.sub(r'<think>.*?</think>', '', raw_expanded_output, flags=re.DOTALL).strip()
                if not expanded_query:
                    expanded_query = question
                retrieval_query = f"{question}\n{expanded_query}"
                print(f"💡 清理後的擴展查詢: {expanded_query}")
            except Exception as e:
                print(f"💡 查詢擴展失敗: {e}, 使用原始查詢。")
                retrieval_query = question

            final_context, all_source_docs = self._get_rag_context(question, retrieval_query)
            formatted_prompt = self.main_prompt.format(context=final_context, question=question)
            return self.stream_and_save(question, formatted_prompt, all_source_docs)
        else:
            print("💬 走通用對話路徑...")
            return self.stream_and_save(question, question, [])

    def stream_and_save(self, question, prompt, source_documents):
        full_answer = ""
        try:
            if source_documents:
                source_data = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in source_documents]
                yield f"data: {json.dumps({'type': 'sources', 'data': source_data})}\n\n"

            for chunk in self.llm.stream(prompt):
                full_answer += chunk
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
            
            self.save_qa(question, full_answer)
        except Exception as e:
            error_msg = f"抱歉，處理您的請求時發生錯誤: {e}"
            print(f"❌ 在串流生成過程中發生錯誤: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
        
        finally:
            # Ensure the temp file is always removed
            #if os.path.exists(file_path):
            #    os.remove(file_path)
                yield f"data: [DONE]\n\n"

    def save_qa(self, question, answer):
        if not answer or not answer.strip():
            print("   -> 偵測到空回答，跳過儲存。")
            return
            
        qa_pair_content = f"問題: {question}\n回答: {answer}"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = { "source": "conversation", "timestamp": current_time }
        new_doc = Document(page_content=qa_pair_content, metadata=metadata)
        self.vector_db.add_documents([new_doc])
        self.update_ensemble_retriever(new_docs=[new_doc])
        print("   -> 對話歷史儲存並同步至混合索引完畢！")
