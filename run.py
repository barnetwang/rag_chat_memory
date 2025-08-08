from app import create_app
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = create_app()
if __name__ == "__main__":
    try:
        from waitress import serve
        import platform
        import socket
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
        except:
            local_ip = "127.0.0.1"
        if platform.system() == "Windows":
            timeout_seconds = 1800
            logging.info("KAIZEN:檢測到 Windows 環境，將使用 Waitress 作為 WSGI 伺服器。")
            logging.info(f"KAIZEN: 連線閒置超時已設定為 {timeout_seconds} 秒。")
            print("\n伺服器已啟動！請在瀏覽器中打開以下任一地址：")
            print(f"  - 本機存取: http://127.0.0.1:5000")
            print(f"  - 區域網路存取: http://{local_ip}:5000\n")
            serve(
                app, 
                host='0.0.0.0', 
                port=5000,
                threads=16, # 適度增加執行緒數量以提高併發能力
                channel_timeout=timeout_seconds # 傳入超時參數
            )
        else:
            logging.info("KAIZEN:在非 Windows 環境下，建議使用 Gunicorn 啟動。現在將使用 Flask 內建伺-服器。")
            logging.info("KAIZEN:現在將使用 Flask/FastAPI 內建伺服器，此伺服器可能沒有嚴格的超時限制，但性能較低。")
            print("\n伺服器已啟動！請在瀏覽器中打開以下任一地址：")
            print(f"  - 本機存取: http://127.0.0.1:5000")
            print(f"  - 區域網路存取: http://{local_ip}:5000\n")
            app.run(host='0.0.0.0', port=5000, debug=False)

    except ImportError:
        logging.warning("KAIZEN: Waitress 未安裝，將使用 Flask/FastAPI 內建的開發伺服器。")
        logging.warning("KAIZEN: 對於長時間運行的任務，內建伺服器可能會不穩定或性能不佳。建議執行 'pip install waitress'")
        #print("請在瀏覽器中打開 http://127.0.0.1:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)