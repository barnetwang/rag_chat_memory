import os
from flask import Flask
from config import Config
from .services import ConversationalRAG, get_ollama_models


rag_chat = None
AVAILABLE_MODELS = []

def create_app(config_class=Config):
    """
    應用程式工廠函式 (Application Factory)
    """
    global rag_chat, AVAILABLE_MODELS

    app = Flask(__name__)
    app.config.from_object(config_class)
    
    print("--- 正在啟動整合 RAG 伺服器 v2.0 (多路徑與混合檢索) ---")
    

    AVAILABLE_MODELS = get_ollama_models(app.config.get('OLLAMA_BASE_URL'))
    
    llm_model = app.config['DEFAULT_MODEL']
    if not AVAILABLE_MODELS:
        print("⚠️ 警告: 沒有從 Ollama 獲取到任何模型，聊天功能將不可用。")
        llm_model = None
    elif llm_model not in AVAILABLE_MODELS:
        print(f"⚠️ 警告: 預設模型 {llm_model} 不可用，將使用第一個可用模型: {AVAILABLE_MODELS[0]}")
        llm_model = AVAILABLE_MODELS[0]


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
        
        print("✅ RAG 核心服務已就緒。")
    else:
        print("❌ RAG 核心服務初始化失敗：無可用 LLM 模型。")



    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    print("--- 伺服器正在運行 ---")
    return app
