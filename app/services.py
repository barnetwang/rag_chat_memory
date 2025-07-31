import os
from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify, Response, render_template
# CORRECTED: Cleaned up imports
from . import rag_chat, AVAILABLE_MODELS

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/api/models', methods=['GET'])
def get_models_and_settings():
    if not rag_chat:
        return jsonify({"error": "RAG service not initialized"}), 503
    return jsonify({
        "models": AVAILABLE_MODELS, 
        "current_model": rag_chat.current_llm_model,
        "history_enabled": rag_chat.use_history,
        "wikipedia_enabled": rag_chat.use_wikipedia,
        "scraper_enabled": rag_chat.use_scraper
    })

@main.route('/api/set_model', methods=['POST'])
def set_model():
    data = request.get_json(); model_name = data.get('model')
    if not model_name or model_name not in AVAILABLE_MODELS: return jsonify({"success": False, "error": "無效或不可用的模型名稱"}), 400
    if rag_chat.set_llm_model(model_name): return jsonify({"success": True})
    return jsonify({"success": False, "error": "伺服器切換模型時發生內部錯誤"}), 500

@main.route('/api/set_history', methods=['POST'])
def set_history():
    data = request.get_json(); enabled = data.get('enabled')
    if not isinstance(enabled, bool): return jsonify({"success": False, "error": "無效的參數"}), 400
    rag_chat.set_history_retrieval(enabled); return jsonify({"success": True})

@main.route('/api/set_wikipedia', methods=['POST'])
def set_wikipedia():
    data = request.get_json(); enabled = data.get('enabled')
    if not isinstance(enabled, bool): return jsonify({"success": False, "error": "無效的參數"}), 400
    rag_chat.set_wikipedia_search(enabled); return jsonify({"success": True})

@main.route('/api/set_scraper', methods=['POST'])
def set_scraper():
    data = request.get_json(); enabled = data.get('enabled')
    if not isinstance(enabled, bool): return jsonify({"success": False, "error": "無效的參數"}), 400
    rag_chat.set_scraper_search(enabled); return jsonify({"success": True})
# ---

@main.route('/ask', methods=['GET'])
def handle_ask():
    question = request.args.get('question')
    if not question: return Response("Error: No question provided", status=400)
    if not rag_chat or not rag_chat.llm: return Response("Error: LLM not available", status=503)
    return Response(rag_chat.ask(question, stream=True), mimetype='text/event-stream')

@main.route('/api/records', methods=['GET'])
def get_records():
    if not rag_chat:
        return jsonify({"error": "RAG service not initialized"}), 503
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        search_query = request.args.get('query', "", type=str)
        paginated_data = rag_chat.search_records(
            query=search_query,
            page=page,
            per_page=per_page
        )
        
        return jsonify(paginated_data)
        
    except Exception as e:
        print(f"❌ 獲取紀錄時發生錯誤: {e}")
        return jsonify({"error": str(e)}), 500

@main.route('/api/upload_document', methods=['POST'])
def upload_document():
    if not rag_chat: return jsonify({"success": False, "error": "RAG 服務未初始化"}), 503
    if 'file' not in request.files: return jsonify({"success": False, "error": "請求中未包含檔案"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"success": False, "error": "未選取檔案"}), 400

    upload_folder = 'uploads' 
    os.makedirs(upload_folder, exist_ok=True)
    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_folder, filename)
    
    try:
        file.save(file_path)
        # CORRECT LOGIC: Call the add_document service method
        rag_chat.add_document(file_path)
        return jsonify({"success": True, "message": f"文件 '{filename}' 已成功處理並加入索引。"})
    except Exception as e:
        print(f"❌ 上傳文件時發生嚴重錯誤: {e}")
        # Clean up the file if something went wrong during processing
        return jsonify({"success": False, "error": f"處理文件時發生錯誤: {str(e)}"}), 500

@main.route('/api/delete', methods=['POST'])
def delete_record():
    if not rag_chat: return jsonify({"success": False, "error": "RAG 服務未初始化"}), 503
        
    data = request.get_json()
    doc_id = data.get('id')
    if not doc_id: return jsonify({"success": False, "error": "請求中缺少 ID"}), 400
        
    try:
        rag_chat.vector_db.delete([doc_id])
        print(f"✅ 成功從向量資料庫刪除 ID: {doc_id}。")
        
        print("🔄 刪除後觸發索引重建...")
        rag_chat.update_ensemble_retriever(full_rebuild=True)
        return jsonify({"success": True, "message": f"成功刪除 ID: {doc_id} 且索引已同步。"})
    except Exception as e:
        return jsonify({"success": False, "error": f"刪除時發生錯誤: {e}"}), 500
@main.route('/api/rebuild_index', methods=['POST'])
def rebuild_index():
    if not rag_chat: return jsonify({"success": False, "error": "RAG 服務未初始化。"}), 503
    try:
        print("🚀 收到手動重建索引的請求...")
        rag_chat.update_ensemble_retriever(full_rebuild=True)
        return jsonify({"success": True, "message": "混合檢索器索引已根據資料庫完整重建。"})
    except Exception as e:
        return jsonify({"success": False, "error": f"重建時发生错误: {e}"}), 500

@main.route('/favicon.ico')
def favicon():
    return '', 204
