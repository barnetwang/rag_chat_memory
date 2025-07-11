import os
import json
import re
import requests
import wikipedia

from datetime import datetime
from bs4 import BeautifulSoup
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_ollama_models(ollama_base_url="http://localhost:11434"):
    try:
        response = requests.get(f"{ollama_base_url}/api/tags")
        response.raise_for_status()
        models_data = response.json().get("models", [])
        return [model["name"] for model in models_data]
    except requests.exceptions.ConnectionError:
        print(f"❌ 錯誤：無法連接到 Ollama 服務 ({ollama_base_url})。請確認 Ollama 正在運行。")
        return []
    except Exception as e:
        print(f"❌ 獲取 Ollama 模型時發生錯誤: {e}")
        return []

class ConversationalRAG:
    def __init__(self, persist_directory, embedding_model_name, llm_model, ollama_base_url, 
                 use_wikipedia=True, use_history=True, use_scraper=True, 
                 history_summary_threshold=2000):
        self.persist_directory = persist_directory
        self.use_wikipedia = use_wikipedia
        self.use_history = use_history
        self.use_scraper = use_scraper
        self.ollama_base_url = ollama_base_url
        self.history_summary_threshold = history_summary_threshold
        
        print("正在初始化 Embedding 模型...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cpu'})

        print("正在初始化/載入向量資料庫...")
        if not os.path.exists(self.persist_directory):
            print("找不到現有資料庫，將創建一個新的。")
            dummy_doc = Document(page_content="start", metadata={"source": "initialization"})
            self.vector_db = Chroma.from_documents([dummy_doc], self.embeddings, persist_directory=self.persist_directory)
        else:
            print("找到現有資料庫，正在載入...")
            self.vector_db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        
        self.llm = None
        self.current_llm_model = None
        self.set_llm_model(llm_model)

        self.retriever = self.vector_db.as_retriever(search_kwargs={'k': 3}) # 增加檢索數量以提供更多來源

        self.main_prompt = PromptTemplate(
            template="""你是一個多功能 AI 助理。請根據以下提供的多種資料來回答使用者的問題。

1.  **[網頁內容]**: 如果提供，這是從使用者指定的網址中抓取的即時內容。請優先使用此內容來回答。
2.  **[維基百科資料]**: 如果沒有提供網頁內容，或網頁內容不足，請使用此資料來回答事實性問題。
3.  **[相關歷史對話]**: 用它來理解問題的上下文，維持對話的連貫性。

請依照以下優先級使用資料: [網頁內容] > [維基百科資料] > 你的內部知識。
如果提供的資料都無法回答，請告知使用者你找不到相關資訊。
在你的最終答案前，你可以使用 <think>...</think> 標籤來寫下你的思考過程，這部分將會被前端介面自動摺疊。

---
[網頁內容]:
{web_context}
---
[維基百科資料]:
{wiki_context}
---
[相關歷史對話]:
{history_context}
---

[使用者當前問題]: {question}

你的回答:""",
            input_variables=["web_context", "wiki_context", "history_context", "question"]
        )
        
        self.summarizer_prompt = PromptTemplate(
            template="請將以下提供的文字內容總結成一段簡潔、流暢的摘要，保留其核心資訊。文字內容如下：\n\n---\n{text_to_summarize}\n---\n\n摘要:",
            input_variables=["text_to_summarize"]
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

    def add_document(self, file_path: str):
        print(f"📄 正在處理新文件: {file_path}")
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        self.vector_db.add_documents(splits)
        print(f"✅ 文件 '{os.path.basename(file_path)}' 已成功加入資料庫。")

        if os.path.exists(file_path):
            os.remove(file_path)

    def _scrape_webpage_text(self, url: str):
        print(f"🕸️ (爬蟲) 正在嘗試爬取網址: {url}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'aside']):
                element.decompose()
            text = '\n'.join(line.strip() for line in soup.get_text().splitlines() if line.strip())
            print(f"✅ (爬蟲) 成功獲取網頁文字，長度: {len(text)} 字元。")
            return text[:4000]
        except Exception as e:
            print(f"❌ (爬蟲) 失敗: {e}")
            return f"無法爬取網址，錯誤: {e}"
            
    def _search_wikipedia(self, query: str):
        print(f"🔍 (外部) 正在從維基百科搜尋 '{query}'...")
        try:
            wikipedia.set_lang("zh-tw")
            suggestion = wikipedia.suggest(query)
            search_query = suggestion if suggestion else query
            summary = wikipedia.summary(search_query, sentences=5)
            print(f"✅ (外部) 找到維基百科摘要。")
            return summary
        except wikipedia.exceptions.PageError:
            return "無相關資料"
        except Exception as e:
            print(f"❌ (外部) 維基百科查詢時發生錯誤: {e}")
            return "查詢時發生錯誤"

    def _summarize_text(self, text: str) -> str:
        print(f"📝 (內部) 正在總結文字，原始長度: {len(text)}")
        try:
            prompt_value = self.summarizer_prompt.format(text_to_summarize=text)
            summary = self.llm.invoke(prompt_value)
            print(f"✅ (內部) 總結完成，新長度: {len(summary)}")
            return summary
        except Exception as e:
            print(f"❌ (內部) 總結時發生錯誤: {e}")
            return text[:self.history_summary_threshold]

    def ask(self, question: str, stream: bool = False):
        print(f"\n🤔 收到請求，問題: '{question}' (流式: {stream})")
        
        web_context = "無"
        if self.use_scraper:
            url_match = re.search(r'https?://[\S]+', question)
            if url_match:
                url = url_match.group(0)
                web_context = self._scrape_webpage_text(url)
        else:
            print("ⓘ (外部) 網頁爬蟲已停用。")

        wiki_context = "維基百科查詢已停用"
        if self.use_wikipedia:
            wiki_context = self._search_wikipedia(question)
        else:
            print("ⓘ (外部) 維基百科查詢已停用。")
        
        history_context = "歷史對話檢索已停用"
        retrieved_docs = []
        if self.use_history:
            print("🔍 (內部) 正在檢索歷史對話...")
            retrieved_docs = self.retriever.get_relevant_documents(question)

            retrieved_docs = [doc for doc in retrieved_docs if doc.page_content != "start"]
            
            if not retrieved_docs:
                history_context = "無相關歷史對話"
            else:
                context_from_docs = "\n---\n".join([doc.page_content for doc in retrieved_docs])
                if len(context_from_docs) > self.history_summary_threshold:
                    print(f"ⓘ (內部) 歷史對話過長 ({len(context_from_docs)} 字元)，正在進行總結...")
                    history_context = self._summarize_text(context_from_docs)
                else:
                    history_context = context_from_docs
        else:
            print("ⓘ (內部) 歷史對話檢索已停用。")

        print("📝 正在組合 Prompt...")
        formatted_prompt = self.main_prompt.format(
            web_context=web_context,
            wiki_context=wiki_context,
            history_context=history_context,
            question=question
        )
        
        if stream:

            return self.stream_and_save(question, formatted_prompt, retrieved_docs)
        else:
            try:
                answer = self.llm.invoke(formatted_prompt)
                self.save_qa(question, answer)
                return answer
            except Exception as e:
                error_msg = f"抱歉，處理您的請求時發生錯誤: {e}"
                print(f"❌ 在非串流生成過程中發生錯誤: {e}")
                return error_msg

    def stream_and_save(self, question, prompt, source_documents):
        full_answer = ""
        try:

            if source_documents:
                source_data = [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata 
                    } 
                    for doc in source_documents
                ]
                yield f"data: {json.dumps({'type': 'sources', 'data': source_data})}\n\n"


            for chunk in self.llm.stream(prompt):
                full_answer += chunk
                response_chunk = {"type": "content", "content": chunk, "error": None}
                yield f"data: {json.dumps(response_chunk)}\n\n"
            
            print("💾 正在儲存本次問答...")
            self.save_qa(question, full_answer)

        except Exception as e:
            error_msg = f"抱歉，處理您的請求時發生錯誤: {e}"
            print(f"❌ 在串流生成過程中發生錯誤: {e}")
            response_chunk = {"type": "error", "error": error_msg}
            yield f"data: {json.dumps(response_chunk)}\n\n"
        
        finally:
            yield f"data: [DONE]\n\n"

    def save_qa(self, question, answer):
        if not answer or answer.strip() == "":
            print("   -> 偵測到空回答，跳過儲存。")
            return
            
        qa_pair_content = f"問題: {question}\n回答: {answer}"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = { "source": "conversation", "timestamp": current_time }
        new_doc = Document(page_content=qa_pair_content, metadata=metadata)
        self.vector_db.add_documents([new_doc])
        print("   -> 對話歷史儲存完畢！")
