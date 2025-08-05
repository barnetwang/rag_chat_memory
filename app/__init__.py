from flask import Flask
from config import Config
from .services import ConversationalRAG, get_ollama_models


rag_chat = None
AVAILABLE_MODELS = []

def create_app(config_class=Config):
    """
    應用程式工廠函數 (Application Factory)。
    創建、配置並初始化應用所需的所有服務。
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    global rag_chat, AVAILABLE_MODELS

    print("--- 正在啟動整合 RAG 伺服器 ---")

    try:
        AVAILABLE_MODELS = get_ollama_models(app.config.get('OLLAMA_BASE_URL'))
        llm_model = app.config['DEFAULT_MODEL']

        if not AVAILABLE_MODELS:
            print("⚠️ 警告: 沒有從 Ollama 獲取到任何模型，RAG 核心服務將不會啟動。")
            llm_model = None
        elif llm_model not in AVAILABLE_MODELS:
            original_model = llm_model
            llm_model = AVAILABLE_MODELS[0]
            print(f"⚠️ 警告: 預設模型 {original_model} 不可用，將自動使用第一個可用模型: {llm_model}")

        if llm_model:
            rag_config = {
                'PERSIST_DIRECTORY': app.config['PERSIST_DIRECTORY'],
                'EMBEDDING_MODEL_NAME': app.config['EMBEDDING_MODEL_NAME'],
                'OLLAMA_BASE_URL': app.config['OLLAMA_BASE_URL'],
                'llm_model': llm_model,
                'CHUNK_SIZE': app.config['CHUNK_SIZE'],
                'CHUNK_OVERLAP': app.config['CHUNK_OVERLAP'],
                'VECTOR_SEARCH_K': app.config['VECTOR_SEARCH_K'],
                'BM25_SEARCH_K': app.config['BM25_SEARCH_K'],
                'ENSEMBLE_WEIGHTS': app.config['ENSEMBLE_WEIGHTS'],
                'EMBEDDING_DEVICE': app.config['EMBEDDING_DEVICE']
            }
            rag_chat = ConversationalRAG(config=rag_config)
            print(f"✅ RAG 核心服務已就緒 (使用模型: {llm_model})。")
        
            print("註冊應用程式關閉時的清理程序...")
            print("✅ 清理程序已註冊。")
        else:
            rag_chat = None

    except Exception as e:
        print(f"❌ 初始化 RAG 服務時發生未知嚴重錯誤: {e}")
        print("系統將在沒有 RAG 功能的情況下運行。")
        rag_chat = None

    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    print("--- 伺服器配置完成，正在啟動 ---")
    return app