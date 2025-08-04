from app import create_app

app = create_app()
if __name__ == "__main__":
    print(f"🚀 啟動 Web UI 伺服器，預設模型: {app.config.get('DEFAULT_MODEL', '未設置')}")
    print("請在瀏覽器中打開 http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
