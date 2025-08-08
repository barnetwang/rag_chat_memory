import os
import json
import logging
from flask import Blueprint, request, jsonify, Response, render_template
from . import rag_chat, AVAILABLE_MODELS

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/api/models', methods=['GET'])
def get_models_and_settings():
    """獲取可用模型列表和當前設定。"""
    if not rag_chat:
        return jsonify({"error": "RAG service not initialized"}), 503
    return jsonify({
        "models": AVAILABLE_MODELS, 
        "current_model": rag_chat.current_llm_model,
        "web_search_enabled": rag_chat.use_web_search
    })

@main.route('/api/set_model', methods=['POST'])
def set_model():
    """設定當前使用的 LLM 模型。"""
    if not rag_chat: return jsonify({"success": False, "error": "RAG service not initialized"}), 503
    data = request.get_json()
    model_name = data.get('model')
    if not model_name or model_name not in AVAILABLE_MODELS:
        return jsonify({"success": False, "error": "無效或不可用的模型名稱"}), 400
    if rag_chat.set_llm_model(model_name):
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "伺服器切換模型時發生內部錯誤"}), 500

@main.route('/api/set_web_search', methods=['POST'])
def set_web_search():
    """啟用或停用網路研究功能。"""
    if not rag_chat:
        return jsonify({"success": False, "error": "RAG service not initialized"}), 503
    enabled = request.json.get('enabled', False)
    rag_chat.set_web_search(enabled)
    return jsonify({"success": True, "message": f"Web search set to {enabled}"})

@main.route('/ask', methods=['GET'])
def handle_ask():
    """處理主要的問答請求，以事件流形式返回結果。"""
    question = request.args.get('question')
    task_id = request.args.get('task_id')
    bypass_assessment = request.args.get('bypass_assessment', 'false').lower() == 'true'

    if not question:
        # 對於 API 錯誤，回傳一個 JSON 會更標準
        error_message = json.dumps({"type": "error", "content": "Error: No question provided"})
        return Response(f"data: {error_message}\n\n", status=400, mimetype='text/event-stream')

    if not rag_chat or not rag_chat.llm:
        error_message = json.dumps({"type": "error", "content": "Error: RAG service not initialized or LLM not available."})
        return Response(f"data: {error_message}\n\n", status=503, mimetype='text/event-stream')

    # 1. 從 services.py 的 ask 函式獲取生成器物件
    generator = rag_chat.ask(
        question=question, 
        stream=True, 
        task_id=task_id, 
        bypass_assessment=bypass_assessment
    )
    
    # 2. 將生成器包裝成 Response 物件，並設定正確的 MIME 類型
    #    這個 return 語句現在是此函式唯一的、正常的出口
    return Response(generator, mimetype='text/event-stream')

@main.route('/api/cancel_task', methods=['POST'])
def cancel_task():
    """取消一個正在運行的深度研究任務。"""
    if not rag_chat:
        return jsonify({"success": False, "error": "RAG service not initialized"}), 503
    data = request.get_json()
    task_id = data.get('task_id')
    if not task_id:
        return jsonify({"success": False, "error": "請求中缺少 task_id"}), 400
    if task_id in rag_chat.active_tasks:
        rag_chat.active_tasks[task_id]['is_cancelled'] = True
        logging.info(f"✅ 收到取消請求，已標記 Task ID: {task_id}")
        return jsonify({"success": True, "message": f"任務 {task_id} 已標記為取消。"})
    else:
        logging.warning(f"⚠️ 收到對一個不存在或已完成的任務的取消請求: {task_id}")
        return jsonify({"success": False, "error": "任務不存在或已完成。"}), 404

# --- [全新] History API Endpoints ---

@main.route('/api/history', methods=['GET'])
def get_history():
    """獲取所有歷史紀錄的列表。"""
    if not rag_chat: return jsonify({"error": "RAG service not initialized"}), 503
    history_list = rag_chat.get_history_list()
    return jsonify(history_list)

@main.route('/api/history/<int:entry_id>', methods=['GET'])
def get_history_entry(entry_id):
    """根據 ID 獲取單條歷史紀錄的完整報告。"""
    if not rag_chat: return jsonify({"error": "RAG service not initialized"}), 503
    entry = rag_chat.get_history_entry(entry_id)
    if entry:
        return jsonify(entry)
    return jsonify({"error": "找不到該歷史紀錄"}), 404

@main.route('/api/history', methods=['POST'])
def add_history_entry():
    """新增一條歷史紀錄。"""
    if not rag_chat: return jsonify({"error": "RAG service not initialized"}), 503
    data = request.get_json()
    title = data.get('title')
    report = data.get('report')
    if not title or not report:
        return jsonify({"error": "請求中缺少 'title' 或 'report'"}), 400
    
    entry_id = rag_chat.add_history_entry(title, report)
    return jsonify({"success": True, "id": entry_id}), 201

@main.route('/api/history/<int:entry_id>', methods=['DELETE'])
def delete_history_entry(entry_id):
    """刪除一條歷史紀錄。"""
    if not rag_chat: return jsonify({"error": "RAG service not initialized"}), 503
    if rag_chat.delete_history_entry(entry_id):
        return jsonify({"success": True})
    return jsonify({"error": "找不到該歷史紀錄或刪除失敗"}), 404

@main.route('/favicon.ico')
def favicon():
    return '', 204
