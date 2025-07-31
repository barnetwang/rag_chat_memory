from flask import Flask
from flask_restx import Api
from app import create_app
from config import Config

app = create_app(Config)
api = Api(app, version='1.0', title='RAG API', description='API for RAG Web UI')

if __name__ == "__main__":
    print(f"🚀 啟動 Web UI 伺服器，預設模型: {app.config['DEFAULT_MODEL']}")
    print("請在瀏覽器中打開 http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
