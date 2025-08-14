# KAIZEN 深度研究報告生成器

KAIZEN 是一款基於大型語言模型（LLM）的智慧應用程式，旨在自動化針對複雜主題的深度研究和報告撰寫過程。它透過一個多階段的智慧代理工作流，從問題拆解、網路研究、資訊綜合到最終報告生成，為使用者提供結構清晰、內容詳實的研究報告。

## ✨ 功能特性

- **🤖 多代理工作流 (Multi-Agent Workflow):** 採用一系列分工明確的 AI 代理（如研究策略師、內容分析師、報告架構師），模擬專家團隊的研究流程。
- **🌐 智慧網路研究 (Intelligent Web Research):** 自動將研究主題分解為具體的搜尋查詢，使用 `DuckDuckGo` 進行搜尋，並利用 `Playwright` 爬取和解析網頁內容。
- **🔍 兩階段內容過濾 (Two-Layer Content Filtering):**
    1.  **標題過濾:** AI 初步評估搜尋結果的相關性。
    2.  **內容審查:** AI 深入閱讀初篩後的網頁全文，確保資訊的真實相關性。
- **📝 結構化報告生成 (Structured Report Generation):**
    1.  **生成大綱:** 在深度研究後，AI 會先產出一個結構化的報告大綱（JSON 格式）供使用者預覽。
    2.  **分章撰寫:** 使用者確認大綱後，AI 會根據大綱和研究資料，逐章撰寫詳細報告。
- **📈 事實校驗 (Fact-Checking):** 報告完成後，系統會自動從報告中提取關鍵論述，並與原始研究資料進行比對，提供一份基礎的校驗結果。
- **💬 互動式介面 (Interactive UI):**
    - 提供即時的進度更新與流程視覺化。
    - 支援任務中斷。
    - 可切換不同的本地 Ollama 模型。
    - 可選擇是否啟用網路研究功能。
- **📚 歷史紀錄與匯出 (History & Export):**
    - 自動儲存已完成的報告。
    - 可隨時查閱、刪除歷史報告。
    - 支援將報告匯出為 Markdown 檔案。

## 🛠️ 技術棧 (Tech Stack)

- **後端 (Backend):** Flask, Waitress
- **AI 框架 (AI Framework):** LangChain, Langchain-Ollama
- **大型語言模型 (LLM):** 透過 Ollama 支援的本地模型 (如 Llama3, GPT-OSS 等)
- **前端 (Frontend):** Vanilla JavaScript, HTML5, CSS3, Marked.js
- **網路爬蟲 (Web Scraping):** Playwright, BeautifulSoup4, DDGS
- **資料庫 (Database):** SQLite

## 🚀 安裝與啟動

在開始之前，請確保您的系統已安裝 [Ollama](https://ollama.com/) 並已拉取至少一個語言模型。

```bash
# 例如，拉取 Open Ai 模型
ollama pull gpt-oss:20b
```

**安裝步驟:**

1.  **克隆專案 (Clone the repository):**
    ```bash
    git clone https://github.com/your-username/rag-chat-memory.git
    cd rag-chat-memory
    ```

2.  **建立並啟用虛擬環境 (Create and activate a virtual environment):**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **安裝 Playwright 瀏覽器依賴:**
    ```bash
    pip install playwright
    playwright install
    ```

4.  **安裝所有必要的 Python 套件:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **啟動應用程式:**
    ```bash
    python run.py
    ```

    伺服器啟動後，請在瀏覽器中開啟 `http://127.0.0.1:5000`。

## 📖 使用方法

1.  **開啟介面:** 在瀏覽器中開啟 `http://127.0.0.1:5000`。
2.  **選擇模型:** 在右側的「設定」面板中，選擇一個已在 Ollama 中運行的模型。
3.  **提出問題:** 在下方的輸入框中，輸入您想要深入研究的複雜主題（例如：「分析並比較蘋果 M3 晶片與高通 Snapdragon X Elite 的性能、功耗與市場定位」）。
4.  **審查大綱:** 系統會進行第一階段的研究並生成報告大綱。您可以在介面中預覽，並點擊「確認並繼續撰寫」按鈕。
5.  **等待報告生成:** 系統將開始撰寫完整報告，並即時將內容串流至介面。
6.  **下載報告:** 報告完成後，點擊下方的「下載報告 (.md)」按鈕即可儲存。

## 📁 專案結構

```
rag-chat-memory/
├── app/                      # Flask 應用程式核心目錄
│   ├── prompts/              # 存放所有 Prompt 模板
│   ├── static/               # 靜態檔案 (Logo)
│   ├── templates/            # HTML 模板
│   ├── __init__.py           # 應用程式工廠
│   ├── routes.py             # API 路由定義
│   └── services.py           # 核心 RAG 與 Agent 服務邏輯
├── config.py                 # 應用程式設定檔
├── requirements.txt          # Python 依賴列表
├── run.py                    # 應用程式啟動腳本
└── database.db               # SQLite 資料庫檔案
```
