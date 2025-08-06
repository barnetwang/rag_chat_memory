from app import create_app
import logging

app = create_app()
if __name__ == "__main__":
    try:
        from waitress import serve
        import platform
        if platform.system() == "Windows":
            logging.info("KAIZEN:檢測到 Windows 環境，將使用 Waitress 作為 WSGI 伺服器。")
            print("請在瀏覽器中打開 http://127.0.0.1:5000")
            serve(app, host='0.0.0.0', port=5000)
        else:
            logging.info("KAIZEN:在非 Windows 環境下，建議使用 Gunicorn 啟動。現在將使用 Flask 內建伺-服器。")
            print("請在瀏覽器中打開 http://127.0.0.1:5000")
            app.run(host='0.0.0.0', port=5000, debug=False)

    except ImportError:
        logging.warning("KAIZEN:Waitress 未安裝，將使用 Flask 內建的開發伺服器。對於長時間運行的任務，這可能會不穩定。建議 'pip install waitress'")
        print("請在瀏覽器中打開 http://127.0.0.1:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)