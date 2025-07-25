<div align="right">
  <b><a href="#-english-readme">English</a></b> | <b><a href="#-中文說明">中文</a></b>
</div>

<a name="-english-readme"></a>

# Local LLM RAG Web UI - v2.0 (High-Performance Edition)

**A high-performance, scalable, 100% local, and private Retrieval-Augmented Generation (RAG) web interface powered by Ollama and LangChain.**

This project has evolved into an advanced solution for running a conversational AI that can process large documents, learn from its own memory, search Wikipedia, and browse the web, all on your local machine.

<img width="1807" height="1179" alt="image" src="https://github.com/user-attachments/assets/b0f520a6-6422-46d1-aebb-2a4e308ab83c" />


---

## 🌟 Key Features

*   **💻 100% Local & Private:** Runs entirely on your machine using [Ollama](https://ollama.com/). Your models and data stay with you.
*   **🚀 High-Performance Hybrid Search:** Combines keyword search (BM25) and semantic search (vectors) for superior retrieval accuracy. The index updates **incrementally and efficiently**, allowing for the ingestion of large documents without server slowdowns.
*   **📄 Multi-Format Document Uploads:** Upload and process various document formats (PDF, DOCX, TXT, etc.) directly through the UI, powered by the `unstructured` library.
*   **🧠 Scalable & Persistent Memory:** Uses [Chroma](https://www.trychroma.com/) as a local vector database. The system is architected to handle tens of thousands of document chunks gracefully.
*   **🌐 Multi-Source RAG Engine:** The AI can augment its responses using:
    *   **Uploaded Documents:** Your private knowledge base.
    *   **Dialogue History:** Remembers past conversations for context.
    *   **Web Scraper:** Can read content from URLs provided in your questions.
    *   **Wikipedia Search:** Looks up information on Wikipedia to answer factual questions.
*   **🔄 Live Model Switching:** Change the underlying LLM model (any model in Ollama) directly from the UI without restarting the server.
*   **🛠️ Database & Index Management:** View, search, and delete individual records. A dedicated API endpoint allows for manually rebuilding the search index to ensure data consistency.

## 🛠️ Technology Stack

*   **Backend:** Flask
*   **LLM Serving:** Ollama
*   **Orchestration:** LangChain (v0.2+ compatible)
*   **Document Processing:** Unstructured
*   **Search & Retrieval:**
    *   **Vector Database:** ChromaDB
    *   **Keyword Search:** In-memory BM25
    *   **Hybrid Search:** LangChain's EnsembleRetriever
*   **Embeddings:** HuggingFace Sentence Transformers

## ⚙️ Installation & Setup

### Prerequisites

*   Python 3.8+
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
    The default model is configured in `config.py`. Make sure you have it downloaded.
    ```bash
    # Example for the default model "deepseek-r1:8b"
    ollama pull deepseek-r1:8b
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
    *   `PERSIST_DIRECTORY`: Folder where the vector database is stored.
    *   `EMBEDDING_MODEL_NAME`: The sentence-transformer model for embeddings.
    *   `EMBEDDING_DEVICE`: The device to run embeddings on (`"cpu"`, `"cuda"`).

*   **RAG Tuning Parameters:**
    *   `CHUNK_SIZE` & `CHUNK_OVERLAP`: Control how documents are split.
    *   `VECTOR_SEARCH_K` & `BM25_SEARCH_K`: The number of results to retrieve from each search method.
    *   `ENSEMBLE_WEIGHTS`: The weights to assign to vector search vs. keyword search (e.g., `[0.5, 0.5]`).

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

Contributions are welcome! If you have a suggestion or want to fix a bug, please follow the standard Fork and Pull Request workflow.

## ⚖️ License & Acknowledgements

This project is open-source and relies on several third-party packages. Please review their licenses before using this project for commercial purposes. Key dependencies include: LangChain, ChromaDB, Unstructured, HuggingFace, and Wikipedia. This project is for educational and research purposes. The user is solely responsible for compliance with all licenses.

---
<br>

<details>
<summary><b>中文說明 (點擊展開)</b></summary>

<a name="-中文說明"></a>

# 本地端 LLM RAG 整合介面 - v2.0 (高性能版)

**一個高性能、可擴展、100% 本地運行、注重隱私的檢索增強生成 (RAG) 網頁應用程式，由 Ollama 與 LangChain 驅動。**

本專案已進化為一個先進的解決方案，讓您在自己的電腦上運行一個強大的對話式 AI。它能處理大型技術文件、從自身記憶中學習、搜尋維基百科並瀏覽網頁，確保完全的隱私。

<img width="1807" height="1179" alt="image" src="https://github.com/user-attachments/assets/b0f520a6-6422-46d1-aebb-2a4e308ab83c" />

---

## 🌟 核心功能

*   **💻 100% 本地化與隱私:** 完全在您的本機上透過 [Ollama](https://ollama.com/) 運行。您的模型和對話資料絕不會離開您的電腦。
*   **🚀 高性能混合式搜尋:** 結合關鍵詞搜尋 (BM25) 與語義搜尋 (向量)，提供卓越的檢索準確度。搜尋索引採用**高效的增量更新**機制，允許載入大型文件而不會拖慢伺服器。
*   **📄 多格式文件上傳:** 透過 UI 直接上傳並處理多種文件格式（PDF, DOCX, TXT 等），由 `unstructured` 函式庫強力驅動。
*   **🧠 可擴展的持久化記憶:** 使用 [Chroma](https://www.trychroma.com/) 作為本地向量資料庫，整體架構經過優化，能夠輕鬆處理數萬級別的文件分塊。
*   **🌐 多源 RAG 引擎:** AI 可以從多種來源獲取資訊以增強其回答：
    *   **上傳的文件:** 您的私有知識庫。
    *   **對話歷史:** 記得過去的對話，理解上下文。
    *   **網頁爬蟲:** 能夠讀取您在問題中提供的網址內容。
    *   **維基百科搜尋:** 查找維基百科來回答事實性問題。
*   **🔄 即時模型切換:** 直接從網頁介面更換底層的 LLM 模型（任何您在 Ollama 中已安裝的模型），伺服器不中斷。
*   **🛠️ 記憶庫與索引管理:** 可瀏覽、搜尋和刪除單筆紀錄。提供專用的 API 端點，用於手動完整重建搜尋索引，確保資料一致性。

## 🛠️ 技術棧

*   **後端框架:** Flask
*   **LLM 服務:** Ollama
*   **AI 框架:** LangChain (兼容 v0.2+)
*   **文件處理:** Unstructured
*   **搜尋與檢索:**
    *   **向量資料庫:** ChromaDB
    *   **關鍵詞搜尋:** In-memory BM25
    *   **混合式搜尋:** LangChain EnsembleRetriever
*   **嵌入模型:** HuggingFace Sentence Transformers

## ⚙️ 安裝與啟動

### 前置需求

*   Python 3.8+
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
    專案的預設模型可在 `config.py` 中設定，請確保您已下載該模型。
    ```bash
    # 以預設模型 "deepseek-r1:8b" 為例
    ollama pull deepseek-r1:8b
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
    *   `PERSIST_DIRECTORY`: 向量資料庫的儲存資料夾。
    *   `EMBEDDING_MODEL_NAME`: 用於生成嵌入向量的句子轉換器模型。
    *   `EMBEDDING_DEVICE`: 運行嵌入模型的設備 (`"cpu"`, `"cuda"`)。

*   **RAG 調優參數:**
    *   `CHUNK_SIZE` & `CHUNK_OVERLAP`: 控制文件分割的方式。
    *   `VECTOR_SEARCH_K` & `BM25_SEARCH_K`: 從每種搜尋方法中檢索的結果數量。
    *   `ENSEMBLE_WEIGHTS`: 分配給向量搜尋與關鍵詞搜尋的權重 (例如 `[0.5, 0.5]`)。

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

歡迎任何形式的貢獻！如果您有建議或想要修復錯誤，請遵循標準的 Fork 和 Pull Request 工作流程。

## ⚖️ 授權與致謝

本專案為開源專案，其依賴的多個第三方套件擁有各自的授權條款。在將本專案用於商業目的前，請務必詳細閱讀並遵守。主要依賴包括：LangChain、ChromaDB、Unstructured、HuggingFace 與 Wikipedia。本專案僅供教育和研究目的，使用者需自行負責確保所有授權和服務條款的合規性。

</details>
