import os
from flask import Flask
from config import Config
from .services import ConversationalRAG, get_ollama_models

rag_chat = None
AVAILABLE_MODELS = []

def create_app(config_class=Config):
    global rag_chat, AVAILABLE_MODELS

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    print("--- 正在啟動整合 RAG 伺服器，請稍候... ---")
    
    AVAILABLE_MODELS = get_ollama_models(app.config['OLLAMA_BASE_URL'])
    
    if not AVAILABLE_MODELS:
        print("⚠️ 警告: 沒有從 Ollama 獲取到任何模型，將無法啟動聊天功能。")
        llm_model = None
    else:
        llm_model = app.config['DEFAULT_MODEL']
        if llm_model not in AVAILABLE_MODELS:
            print(f"⚠️ 警告: 預設模型 {llm_model} 不可用，將使用第一個可用模型: {AVAILABLE_MODELS[0]}")
            llm_model = AVAILABLE_MODELS[0]

    rag_chat = ConversationalRAG(
        persist_directory=app.config['PERSIST_DIRECTORY'],
        embedding_model_name=app.config['EMBEDDING_MODEL_NAME'],
        llm_model=llm_model,
        ollama_base_url=app.config['OLLAMA_BASE_URL']
    )
    
    print("--- RAG 引擎已就緒，伺服器正在運行 ---")

    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
