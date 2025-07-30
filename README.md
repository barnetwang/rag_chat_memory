<div align="right">
  <b><a href="#-english-readme">English</a></b> | <b><a href="#-中文說明">中文</a></b>
</div>

<a name="-english-readme"></a>

# Local LLM RAG Web UI - v2.5 (Intelligent Engine Edition)

**An intelligent, high-performance, 100% local, and private Retrieval-Augmented Generation (RAG) web interface powered by Ollama and LangChain.**

This project has evolved into a production-ready solution for running a conversational AI. It features an intelligent routing system, advanced hybrid search, and a robust document processing pipeline, all on your local machine.

![image](https://github.com/user-attachments/assets/06047960-b8c0-46f0-9447-c6810934a076)

---

## 🌟 Key Features

*   **💻 100% Local & Private:** Runs entirely on your machine using [Ollama](https://ollama.com/). Your models and data stay with you.
*   **🧠 Intelligent Query Routing:** Employs a multi-path RAG architecture. An LLM-powered router analyzes incoming questions to distinguish between complex RAG queries and general conversation, directing them to the optimal processing pipeline.
*   **🚀 High-Performance Hybrid Search:** Combines keyword search (BM25) and semantic search (vectors) for superior retrieval accuracy on both technical terms and conceptual questions. The keyword index is efficiently updated in memory after new documents are added.
*   **📄 Robust Multi-Format Document Ingestion:** Upload and process various document formats (PDF, DOCX, TXT, etc.). The pipeline uses specialized parsers like `PyMuPDF` for high-quality text extraction from complex layouts and includes an advanced pre-processing step to clean headers, footers, and other noise.
*   **💡 AI-Powered Query Expansion:** Automatically refines vague user queries into more specific, detailed search terms to significantly improve retrieval "hit rates".
*   **📝 Trustworthy Answers with Source Citation:** Every answer generated from the knowledge base is accompanied by clickable source links, allowing users to trace information back to the original document snippets.
*   **🌐 Multi-Source RAG Engine:** Augments responses using:
    *   **Uploaded Documents:** Your private knowledge base.
    *   **Dialogue History:** Remembers past conversations for context.
    *   **Web Scraper & Wikipedia Search:** (Optional) Can be enabled for real-time information.
*   **🔄 Live Model Switching:** Change the underlying LLM (any model in Ollama) directly from the UI.
*   **🛠️ Database & Index Management:** View, search, and delete individual records. The system is architected to handle tens of thousands of document chunks gracefully with batch processing.

## 🛠️ Technology Stack

*   **Backend:** Flask
*   **LLM Serving:** Ollama
*   **Orchestration:** LangChain (v0.2+)
*   **Document Processing:** PyMuPDF, Unstructured
*   **Search & Retrieval:**
    *   **Vector Database:** ChromaDB
    *   **Keyword Search:** rank_bm25
    *   **Hybrid Search:** LangChain EnsembleRetriever
*   **Embeddings:** HuggingFace Sentence Transformers (e.g., `nomic-embed-text`)

## ⚙️ Installation & Setup

### Prerequisites

*   Python 3.9+
*   Git
*   [Ollama](https://ollama.com/) installed and running.

### Step-by-Step Guide

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/barnetwang/rag-chat-memory.git
    cd rag-chat-memory
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Pull an Ollama model:**
    The default LLM and Embedding models are configured in `config.py`. Make sure you have them downloaded.
    ```bash
    # Example for the default LLM
    ollama pull gemma:2b

    # Example for the default Embedding model (if you choose to use Ollama for embeddings)
    ollama pull nomic-embed-text
    ```

4.  **Launch the application:**
    ```bash
    python run.py
    ```

5.  **Open the Web UI:**
    Open your browser and navigate to **http://127.0.0.1:5000**.

## 🔧 Configuration

You can customize the application's behavior by editing the `config.py` file:

*   **Core Settings:**
    *   `DEFAULT_MODEL`: The default LLM model to use on startup.
    *   `EMBEDDING_MODEL_NAME`: The sentence-transformer model for embeddings from Hugging Face.
    *   `PERSIST_DIRECTORY`: Folder where the vector database is stored.

*   **RAG Tuning Parameters:**
    *   `VECTOR_SEARCH_K` & `BM25_SEARCH_K`: The number of results to retrieve from each search method.
    *   `ENSEMBLE_WEIGHTS`: The weights to assign to vector search vs. keyword search.

## 📂 Project Structure
```
/
├── app/
│   ├── __init__.py      # Initializes the Flask app
│   ├── routes.py        # Contains all Flask routes and API endpoints
│   ├── services.py      # Contains the core ConversationalRAG class
│   ├── static/          # For CSS/JS files
│   └── templates/
│       └── index.html   # The single-page HTML for the frontend UI
├── config.py            # All user-configurable settings
├── run.py               # The main entry point to start the application
├── requirements.txt     # Python package dependencies
└── README.md            # This file
```

## 🤝 Contributing

Contributions are welcome! Please follow the standard Fork and Pull Request workflow.

## ⚖️ License & Acknowledgements

This project is open-source and relies on several third-party packages. Please review their licenses before using this project for commercial purposes.

---
<br>

<details>
<summary><b>中文說明 (點擊展開)</b></summary>

<a name="-中文說明"></a>

# 本地端 LLM RAG 整合介面 - v2.5 (智能引擎版)

**一個智能、高性能、100% 本地運行、注重隱私的檢索增強生成 (RAG) 網頁應用程式，由 Ollama 與 LangChain 驅動。**

本專案已進化為一個生產級的解決方案，讓您在自己的電腦上運行一個強大的對話式 AI。它具備智能路由系統、先進的混合式搜尋、以及穩健的文件處理流程，確保完全的隱私。

![image](https://github.com/user-attachments/assets/06047960-b8c0-46f0-9447-c6810934a076)

---

## 🌟 核心功能

*   **💻 100% 本地化與隱私:** 完全在您的本機上透過 [Ollama](https://ollama.com/) 運行。
*   **🧠 智能查詢路由:** 採用多路徑 RAG 架構。由 LLM 驅動的路由器會分析傳入的問題，區分需要深度檢索的複雜查詢和一般對話，並將它們導向最佳的處理流程。
*   **🚀 高性能混合式搜尋:** 結合關鍵詞搜尋 (BM25) 與語義搜尋 (向量)，在處理技術術語和概念性問題時都能達到卓越的檢索準確度。關鍵詞索引在新增文件後會高效地在記憶體中進行更新。
*   **📄 穩健的多格式文件處理:** 可上傳並處理多種文件格式（PDF, DOCX 等）。處理流程使用如 `PyMuPDF` 等專業解析器，以從複雜佈局中進行高質量的文本提取，並包含先進的預處理步驟來清理頁眉、頁腳等噪音。
*   **💡 AI 驅動的查詢擴展:** 自動將使用者模糊的查詢，細化為更具體、更專業的搜索詞，顯著提升檢索“命中率”。
*   **📝 可信賴的答案與來源引用:** 每個從知識庫生成的回答都會附帶可點擊的來源連結，讓使用者能追溯資訊至原始的文件片段。
*   **🌐 多源 RAG 引擎:** 可整合來自**上傳的文件**、**對話歷史**、**網頁爬蟲**和**維基百科**的多種資訊源。
*   **🔄 即時模型切換:** 直接從 UI 更換底層的 LLM 模型。
*   **🛠️ 記憶庫與索引管理:** 可瀏覽、搜尋和刪除單筆紀錄。系統透過分批處理，能夠輕鬆應對包含數萬片段的大型文件。

## 🛠️ 技術棧

*   **後端框架:** Flask
*   **LLM 服務:** Ollama
*   **AI 框架:** LangChain (v0.2+)
*   **文件處理:** PyMuPDF, Unstructured
*   **搜尋與檢索:**
    *   **向量資料庫:** ChromaDB
    *   **關鍵詞搜尋:** rank_bm25
    *   **混合式搜尋:** LangChain EnsembleRetriever
*   **嵌入模型:** HuggingFace Sentence Transformers (例如 `nomic-embed-text`)

## ⚙️ 安裝與啟動

### 前置需求

*   Python 3.9+
*   Git
*   [Ollama](https://ollama.com/) 已安裝並正在運行。

### 步驟指南

1.  **克隆專案倉庫：**
    ```bash
    git clone https://github.com/barnetwang/rag-chat-memory.git
    cd rag-chat-memory
    ```

2.  **安裝 Python 依賴套件：**
    ```bash
    pip install -r requirements.txt
    ```

3.  **下載 Ollama 模型：**
    專案的預設模型可在 `config.py` 中設定，請確保您已下載。
    ```bash
    # 以 gemma:2b 為例
    ollama pull gemma:2b
    ```

4.  **啟動應用程式：**
    ```bash
    python run.py
    ```

5.  **打開 Web UI：**
    打開您的瀏覽器，並訪問 **http://127.0.0.1:5000**。

## 🔧 專案設定

您可以透過編輯 `config.py` 檔案來自訂應用程式的行為：

*   **核心設定:**
    *   `DEFAULT_MODEL`: 啟動時預設使用的 LLM 模型。
    *   `EMBEDDING_MODEL_NAME`: 從 Hugging Face 下載的嵌入模型。
    *   `PERSIST_DIRECTORY`: 向量資料庫的儲存資料夾。

*   **RAG 調優參數:**
    *   `VECTOR_SEARCH_K` & `BM25_SEARCH_K`: 從每種搜尋方法中檢索的結果數量。
    *   `ENSEMBLE_WEIGHTS`: 分配給向量搜尋與關鍵詞搜尋的權重。

## 📂 專案結構
```
/
├── app/
│   ├── __init__.py      # 初始化 Flask App
│   ├── routes.py        # 包含所有 Flask 路由和 API 端點
│   ├── services.py      # 包含核心的 ConversationalRAG 類別
│   ├── static/          # 用於存放 CSS/JS 檔案
│   └── templates/
│       └── index.html   # 前端 UI 的單頁 HTML 檔案
├── config.py            # 所有使用者可配置的設定
├── run.py               # 啟動應用程式的主要進入點
├── requirements.txt     # Python 依賴套件列表
└── README.md            # 本說明檔案
```

## 🤝 貢獻指南

歡迎任何形式的貢獻！請遵循標準的 Fork 和 Pull Request 工作流程。

## ⚖️ 授權與致謝

本專案為開源專案，其依賴的多個第三方套件擁有各自的授權條款。在將本專案用於商業目的前，請務必詳細閱讀並遵守。

</details>
