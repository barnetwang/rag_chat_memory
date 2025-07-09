<div align="right">
  <b><a href="#-english-readme">English</a></b> | <b><a href="#-中文說明">中文</a></b>
</div>
<a name="-english-readme"></a>
# Local LLM RAG Web UI

**A full-featured, 100% local, and private Retrieval-Augmented Generation (RAG) web interface powered by Ollama and LangChain.**

This project provides a complete, out-of-the-box solution for running a conversational AI that can use its own memory, search Wikipedia, and browse the web, all on your local machine. No data ever leaves your computer.

![image](https://github.com/user-attachments/assets/06047960-b8c0-46f0-9447-c6810934a076)

---

## 🌟 Key Features

*   **💻 100% Local & Private:** Runs entirely on your machine using [Ollama](https://ollama.com/). Your models and data stay with you.
*   **🚀 Full-Featured Web UI:** A clean, modern interface built with Flask for chatting and managing the AI's memory.
*   **🧠 Multi-Source RAG:** The AI can augment its responses using:
    *   **Dialogue History:** Remembers past conversations for context.
    *   **Web Scraper:** Can read content from URLs provided in your questions.
    *   **Wikipedia Search:** Looks up information on Wikipedia to answer factual questions.
*   **🎛️ Dynamic Toggles:** Easily enable or disable the Web Scraper, Wikipedia Search, and Dialogue History on-the-fly from the UI.
*   **🔄 Live Model Switching:** Change the underlying LLM model (any model you have in Ollama) directly from the web interface without restarting the server.
*   **💾 Persistent Memory:** Uses [Chroma](https://www.trychroma.com/) as a local vector database to store and retrieve conversation history across sessions.
*   **🛠️ Database Management:** View, search, and delete individual conversation records from the AI's memory through the UI.

## 🛠️ Technology Stack

*   **Backend:** Flask
*   **LLM Serving:** Ollama
*   **Orchestration:** LangChain
*   **Vector Database:** ChromaDB
*   **Embeddings:** HuggingFace `moka-ai/m3e-base`

## ⚙️ Installation & Setup

Follow these steps to get the application running.

### Prerequisites

*   Python 3.8+
*   Git
*   [Ollama](https://ollama.com/) installed and running.

### Step-by-Step Guide

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/barnetwang/rag_chat_memory.git
    cd rag_chat_memory
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Pull an Ollama model:**
    Make sure you have a model downloaded in Ollama. The default is `gemma3n:e4b`.
    ```bash
    ollama pull gemma3n:e4b
    ```
    *(You can use any other model, like `llama3`, `mistral`, etc.)*

4.  **Launch the application:**
    ```bash
    python run.py
    ```

5.  **Open the Web UI:**
    Open your browser and navigate to **http://127.0.0.1:5000**.

## 🔧 Configuration

You can customize the application's behavior by editing the `config.py` file:

*   `DEFAULT_MODEL`: Change the default LLM model to use on startup.
*   `PERSIST_DIRECTORY`: Change the folder where the vector database is stored.
*   `EMBEDDING_MODEL_NAME`: Specify a different sentence-transformer model for embeddings.

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

Contributions are welcome! If you have a suggestion or want to fix a bug:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5.  Push to the branch (`git push origin feature/AmazingFeature`).
6.  Open a Pull Request.

## ⚖️ License & Acknowledgements

This project is open-source. However, it relies on several third-party packages with their own licenses. Please review them before using this project for commercial purposes.

*   **LangChain:** [MIT License](https://github.com/langchain-ai/langchain/blob/master/LICENSE)
*   **ChromaDB:** [Apache-2.0 License](https://github.com/chroma-core/chroma/blob/main/LICENSE)
*   **HuggingFace Models:** Models like `moka-ai/m3e-base` have their own licenses. Please check the model's page.
*   **Wikipedia API:** Data usage is subject to the [Creative Commons Attribution-ShareAlike License](https://creativecommons.org/licenses/by-sa/3.0/).

This project is for educational and research purposes. The user is solely responsible for compliance with all licenses and terms of service.

---
<br>

<details>
<summary><b>中文說明 (點擊展開)</b></summary>

<a name="-中文說明"></a>
# 本地端 LLM RAG 整合介面

**一個功能完整、100% 本地運行、注重隱私的檢索增強生成 (RAG) 網頁應用程式，由 Ollama 與 LangChain 驅動。**

本專案提供一個開箱即用的解決方案，讓您在自己的電腦上運行一個強大的對話式 AI。它能夠利用自身的記憶、搜尋維基百科、並瀏覽網頁，而您的所有資料和模型都將保留在本地，確保完全的隱私。

![image](https://github.com/user-attachments/assets/06047960-b8c0-46f0-9447-c6810934a076)


---

## 🌟 核心功能

*   **💻 100% 本地化與隱私:** 完全在您的本機上透過 [Ollama](https://ollama.com/) 運行。您的模型和對話資料絕不會離開您的電腦。
*   **🚀 功能完整的 Web UI:** 一個使用 Flask 打造的現代化、乾淨的網頁介面，用於聊天和管理 AI 的記憶庫。
*   **🧠 多源 RAG 引擎:** AI 可以從多種來源獲取資訊以增強其回答：
    *   **對話歷史:** 記得過去的對話，理解上下文。
    *   **網頁爬蟲:** 能夠讀取您在問題中提供的網址內容。
    *   **維基百科搜尋:** 查找維基百科來回答事實性問題。
*   **🎛️ 動態功能開關:** 在 UI 上即時啟用或停用網頁爬蟲、維基百科搜尋和歷史對話功能，無需重啟。
*   **🔄 即時模型切換:** 直接從網頁介面更換底層的 LLM 模型（任何您在 Ollama 中已安裝的模型），伺服器不中斷。
*   **💾 持久化記憶:** 使用 [Chroma](https://www.trychroma.com/) 作為本地向量資料庫，跨會話儲存和檢索對話歷史。
*   **🛠️ 記憶庫管理:** 透過 UI 瀏覽、搜尋和刪除 AI 記憶庫中的每一筆對話紀錄。

## 🛠️ 技術棧

*   **後端框架:** Flask
*   **LLM 服務:** Ollama
*   **AI 框架:** LangChain
*   **向量資料庫:** ChromaDB
*   **嵌入模型:** HuggingFace `moka-ai/m3e-base`

## ⚙️ 安裝與啟動

請依照以下步驟來啟動應用程式。

### 前置需求

*   Python 3.8+
*   Git
*   [Ollama](https://ollama.com/) 已安裝並正在運行。

### 步驟指南

1.  **克隆專案倉庫：**
    ```bash
    git clone https://github.com/barnetwang/rag_chat_memory.git
    cd rag_chat_memory
    ```

2.  **安裝 Python 依賴套件：**
    ```bash
    pip install -r requirements.txt
    ```

3.  **下載 Ollama 模型：**
    請確保您已在 Ollama 中下載了一個模型。專案預設使用 `gemma3n:e4b`。
    ```bash
    ollama pull gemma3n:e4b
    ```
    *(您也可以使用任何其他模型，例如 `llama3`, `mistral` 等)*

4.  **啟動應用程式：**
    ```bash
    python run.py
    ```

5.  **打開 Web UI：**
    打開您的瀏覽器，並訪問 **http://127.0.0.1:5000**。

## 🔧 專案設定

您可以透過編輯 `config.py` 檔案來自訂應用程式的行為：

*   `DEFAULT_MODEL`: 更改啟動時預設使用的 LLM 模型。
*   `PERSIST_DIRECTORY`: 更改向量資料庫的儲存資料夾。
*   `EMBEDDING_MODEL_NAME`: 指定一個不同的句子轉換器模型用於生成嵌入向量。

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

歡迎任何形式的貢獻！如果您有建議或想要修復錯誤：
1.  Fork 本倉庫。
2.  建立一個新的分支 (`git checkout -b feature/AmazingFeature`)。
3.  進行您的修改。
4.  提交您的變更 (`git commit -m 'Add some AmazingFeature'`)。
5.  將分支推送到遠端 (`git push origin feature/AmazingFeature`)。
6.  開啟一個 Pull Request。

## ⚖️ 授權與致謝

本專案為開源專案，但其依賴的多個第三方套件擁有各自的授權條款。在將本專案用於商業目的前，請務必詳細閱讀並遵守各元件的授權。

*   **LangChain:** [MIT License](https://github.com/langchain-ai/langchain/blob/master/LICENSE)
*   **ChromaDB:** [Apache-2.0 License](https://github.com/chroma-core/chroma/blob/main/LICENSE)
*   **HuggingFace 模型:** 如 `moka-ai/m3e-base` 等模型均有其自身的授權，請查閱其模型頁面。
*   **Wikipedia API:** 資料使用需遵守 [Creative Commons Attribution-ShareAlike License](https://creativecommons.org/licenses/by-sa/3.0/)。

本專案僅供教育和研究目的。使用者需自行負責確保所有授權和服務條款的合規性。

</details>
