<div align="right">
  <b><a href="#-english-readme">English</a></b> | <b><a href="#-中文說明">中文</a></b>
</div>

<a name="-english-readme"></a>

# Local LLM RAG Web UI

**An intelligent, high-performance, 100% local, and private Retrieval-Augmented Generation (RAG) web interface powered by Ollama and LangChain.**

This project has evolved beyond a simple RAG UI into a powerful, **autonomous AI research assistant** that runs 100% locally on your machine. It leverages a multi-agent "Expert Team" workflow to transform a single complex question into a detailed, well-researched, and fully-cited report.

Designed for privacy, cost-effectiveness, and professional-grade output, this tool is your personal, automated strategy consultant.

![image](https://github.com/user-attachments/assets/b0f520a6-6422-46d1-aebb-2a4e308ab83c)

---

## 🌟 Core Features

*   **🤖 Autonomous "Expert Team" Workflow**: The system's core innovation. A single user query triggers a sophisticated multi-agent process:
    1.  **Router**: Intelligently identifies complex research tasks.
    2.  **Task Decomposer**: Breaks the main query into 2-4 distinct, non-overlapping research sub-tasks.
    3.  **Web Researcher**: For each sub-task, it generates keywords, scours the web (including HTML and PDFs), scrapes content with an advanced crawler (`Playwright` + `PyMuPDF`), and gathers high-quality data.
    4.  **Blueprint Architect**: Synthesizes all research memos into a structured JSON report blueprint.
    5.  **Chapter Writer**: A dedicated agent writes each chapter of the final report based on the blueprint and the full research context, complete with source citations.
*   **💻 100% Local & Privacy-Focused**: Runs entirely on your machine via [Ollama 🔗](https://ollama.com/), ensuring your data never leaves your computer.
*   **📄 Multi-Format RAG Engine**: Ingests and processes a wide range of document formats (`PDF`, `DOCX`, etc.) and integrates data from **uploaded files**, **chat history**, and **real-time web research**.
*   **🔗 Verifiable & Trustworthy Answers**: Every piece of information in the generated report is tied back to its source. The final output includes a comprehensive list of clickable source links.
*   **🚀 High-Performance Hybrid Search**: (For local documents) Combines BM25 keyword search and vector search to achieve superior retrieval accuracy.
*   **🔄 Real-time Model Switching**: Easily switch the underlying LLM model directly from the UI without restarting the server.
*   **🛠️ Memory & Index Management**: Full control to browse, search, and delete individual records from your knowledge base.

---

## 🛠️ Technology Stack

- **Backend Framework:** [Flask](https://flask.palletsprojects.com/)
- **LLM Service Provider:** [Ollama](https://ollama.com/)
- **AI Framework:** [LangChain (v0.2+)](https://python.langchain.com/docs/)
- **Web Scraping & Automation:** [Playwright](https://playwright.dev/) & [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/)
- **Document Processing:** `PyMuPDF` (for superior PDF extraction), `Unstructured`
- **Search & Retrieval:**
    -   Vector Database: [ChromaDB](https://github.com/chromadb/chroma)
    -   Keyword Search: `rank_bm25`
    -   Hybrid Retrieval: LangChain’s `EnsembleRetriever`
- **Embedding Models:** [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers)

---

## ⚙️ Installation & Setup

### Prerequisites
- Python ≥ 3.9
- Git
- [Ollama](https://ollama.com/) installed and running (`ollama serve`)

### Step-by-Step Guide

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/barnetwang/rag-chatbot.git
    cd rag-chatbot
    ```

2.  **Install Python Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Playwright Browsers** (One-time setup)
    This is crucial for the web research agent.
    ```bash
    playwright install
    ```

4.  **Pull Required Ollama Models**
    Edit `config.py` to set your preferred models, then download them. For example:
    ```bash
    ollama pull qwen3:latest       # Main LLM for generation
    ollama pull nomic-embed-text      # Embedding model
    ```

5.  **Start the Application**
    ```bash
    python run.py
    ```

6.  **Open the Web UI** in your browser at `http://127.0.0.1:5000`

---

## 📂 Project Structure

```
/
├── app/
│ ├── init.py # Initialize Flask app & core services
│ ├── routes.py # API endpoints
│ ├── services.py # Core logic class: ConversationalRAG
│ ├── static/ # Frontend CSS & JS
│ ├── templates/
│ │ └── index.html # Main web interface
│ └── prompts/ # Houses all prompt templates (.txt files) for agents
├── config.py # All customizable settings
├── run.py # Entry point to launch the app
├── requirements.txt # Python dependencies
└── README.md # You're here!
```

---

## 🤝 Contributing

We welcome contributions!
Follow these steps:

1. Fork the repo
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add some feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## ⚖️ License & Acknowledgements

This project is open-source and uses various third-party libraries.

Please review their licenses carefully before using this code for commercial purposes.
---
<br>

<details>
<summary><b>中文說明 (點擊展開)</b></summary>

<a name="-中文說明"></a>

# 本地端 LLM RAG 整合介面

**一個智能、高性能、100% 本地運行、注重隱私的檢索增強生成 (RAG) 網頁應用程式，由 Ollama 與 LangChain 驅動。**

本專案已進化為一個強大的**自主 AI 研究助理**，能夠 100% 在您的本機上運行。它利用一個「專家小組」多智能體工作流，將一個複雜的問題，轉化為一份詳細、研究充分、且引用來源完整的深度報告。

專為注重隱私、成本效益和專業級輸出的使用者設計，此工具是您的個人自動化策略顧問。

![image](https://github.com/user-attachments/assets/b0f520a6-6422-46d1-aebb-2a4e308ab83c)

---

*   **🤖 自主「專家小組」工作流**: 本系統的核心創新。使用者的一個複雜問題將觸發一套精密的自動化流程：
    1.  **路由器 (Router)**: 智能識別需要深度研究的複雜任務。
    2.  **任務拆解器 (Task Decomposer)**: 將主問題分解為 2-4 個獨立、不重複的研究子任務。
    3.  **網路研究員 (Web Researcher)**: 為每個子任務生成關鍵詞，全面搜索網路（包含網頁與PDF），使用先進的爬蟲 (`Playwright` + `PyMuPDF`) 抓取高品質資料。
    4.  **藍圖架構師 (Blueprint Architect)**: 將所有研究備忘錄綜合成一份結構化的 JSON 報告藍圖。
    5.  **章節撰寫器 (Chapter Writer)**: 一個專門的 AI 智能體，根據藍圖和全部研究資料，逐章撰寫最終報告的詳細內容，並附上來源引用。
*   **💻 100% 本地化與隱私保護**: 完全透過 [Ollama](https://ollama.com/) 在您的本機運行，確保您的資料永遠不會離開您的電腦。
*   **📄 多格式 RAG 引擎**: 可讀取並處理多種文件格式 (`PDF`, `DOCX` 等)，並能整合來自**上傳的檔案**、**對話歷史**及**即時網路研究**的資料。
*   **🔗 可驗證與可信賴的答案**: 報告中的每一條資訊都與其來源掛鉤。最終的輸出會包含一份完整的、可點擊的來源連結列表。
*   **🚀 高性能混合式搜尋**: (針對本地文件) 結合關鍵詞搜尋 (BM25) 與語義向量搜尋，實現卓越的檢索準確度。
*   **🔄 即時模型切換**: 無需重啟伺服器，直接從 UI 介面輕鬆切換底層的 LLM 模型。
*   **🛠️ 記憶庫與索引管理**: 完整的功能，讓您能瀏覽、搜尋和刪除知識庫中的單筆紀錄。

---

## 🛠️ 技術棧

*   **後端框架:** [Flask](https://flask.palletsprojects.com/)
*   **LLM 服務:** [Ollama](https://ollama.com/)
*   **AI 框架:** [LangChain (v0.2+)](https://python.langchain.com/docs/)
*   **文件處理:** `PyMuPDF`, `Unstructured`
*   **搜尋與檢索:**
    *   **向量資料庫:** [ChromaDB](https://github.com/chromadb/chroma)
    *   **關鍵詞搜尋:** [`rank_bm25`](https://pypi.org/project/rank-bm25/)
    *   **混合式搜尋:** LangChain 的 [`EnsembleRetriever`](https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble/)
*   **嵌入模型:** [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers)
  - 預設示例: `nomic-embed-text`

---

## ⚙️ 安裝與啟動

### 前置條件

確保以下項目均已安裝並執行:

- Python ≥ 3.9
- Git
- [Ollama](https://ollama.com/) 已安裝且運行中 (`ollama serve`)

> 💡 **提示**: 使用 `screen` 或 `nohup` 在背景啟動 Ollama，避免終端被佔用。

---

### 操作步驟

1. **複製倉儲**
   ```bash
   git clone https://github.com/barnetwang/rag-chatbot.git
   cd rag-chatbot
   ```

2. **安裝依賴套件**
   ```bash
   pip install -r requirements.txt
   ```

3. **下載所需模型**

   修改 `config.py` 中的預設模型設定後進行下載：
   ```bash
   ollama pull qwen3:latest          # 默認 LLM 模型
   oollama pull nomic-embed-text  # 默認嵌入模型（如需使用）
   ```

4. **啟動應用程序**
   ```bash
   python run.py
   ```

5. **開啟介面**
   瀏覽器中輸入：
   ```
   http://127.0.0.1:5000
   ```

---

## 🔧 配置檔案說明

修改 `config.py` 中的參數進行自訂：

### 核心設定

| 參數名稱              | 說明 |
|-----------------------|------|
| `DEFAULT_MODEL`       | 啟動時預設使用的 LLM 模型 |
| `EMBEDDING_MODEL_NAME`| HuggingFace 中的嵌入式模型名稱 |
| `PERSIST_DIRECTORY`   | ChromaDB 向量儲存路徑 |

### 搜尋調整參數

| 參數名稱              | 說明 |
|-----------------------|------|
| `VECTOR_SEARCH_K`     | 向量搜尋返回結果數目 |
| `BM25_SEARCH_K`       | BM25 搜尋返回結果數目 |
| `ENSEMBLE_WEIGHTS`    | 向量與關鍵字搜索結果合成時的權重分配比例 |

---

## 📂 專案架構

```
/
├── app/
│ ├── init.py      # Flask 初始化
│ ├── routes.py        # API 端點定義
│ ├── services.py      # 主要業務邏輯：ConversationalRAG 類別
│ ├── static/          # 前端 CSS & JS 資料夾
│ ├── templates/
│ │ └── index.html       # 單頁式介面 HTML 檔案
│ └── prompts/           # 存放所有代理程式的提示範本（.txt 檔案）
├── config.py            # 可變動設定檔
├── run.py               # 啟動腳本入口點
├── requirements.txt     # 所需 Python 套件清單
└── README.md            # 說明文件（就是這個！）
```

---

## 🤝 貢獻指南

歡迎任何形式的貢獻！請遵循標準的 Fork 和 Pull Request 工作流程。

## ⚖️ 授權與致謝

本專案為開源專案，其依賴的多個第三方套件擁有各自的授權條款。在將本專案用於商業目的前，請務必詳細閱讀並遵守。

</details>
