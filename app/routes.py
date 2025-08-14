import os
import json
import logging
import base64
import binascii
import urllib.parse
from flask import Blueprint, request, jsonify, Response, render_template
from . import rag_chat, AVAILABLE_MODELS

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/api/models', methods=['GET'])
def get_models_and_settings():
    print(f"--- API HIT: /api/models ---")
    print(f"AVAILABLE_MODELS: {AVAILABLE_MODELS}")
    print(f"rag_chat is None: {rag_chat is None}")

    if not AVAILABLE_MODELS:
        print("--- API RESPONSE: No models available (503) ---")
        return jsonify({
            "error": "無法獲取模型列表。請確認 Ollama 服務正在運行，並且至少已拉取一個模型 (例如: ollama pull llama3)。"
        }), 503

    if not rag_chat:
        print("--- API RESPONSE: RAG not initialized (503) ---")
        return jsonify({
            "error": "RAG 服務未初始化，但模型列表可用。請檢查後端日誌。"
        }), 503

    response_data = {
        "models": AVAILABLE_MODELS,
        "current_model": rag_chat.current_llm_model,
        "web_search_enabled": rag_chat.use_web_search
    }
    print(f"--- API RESPONSE: Success (200), Data: {response_data} ---")
    return jsonify(response_data)

@main.route('/api/set_model', methods=['POST'])
def set_model():
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
    if not rag_chat:
        return jsonify({"success": False, "error": "RAG service not initialized"}), 503
    enabled = request.json.get('enabled', False)
    rag_chat.set_web_search(enabled)
    return jsonify({"success": True, "message": f"Web search set to {enabled}"})

@main.route('/ask', methods=['GET'])
def handle_ask():
    question = request.args.get('question')
    task_id = request.args.get('task_id')
    bypass_assessment = request.args.get('bypass_assessment', 'false').lower() == 'true'

    if not question:
        error_message = json.dumps({"type": "error", "content": "Error: No question provided"})
        return Response(f"data: {error_message}\n\n", status=400, mimetype='text/event-stream')

    if not rag_chat or not rag_chat.llm:
        error_message = json.dumps({"type": "error", "content": "Error: RAG service not initialized or LLM not available."})
        return Response(f"data: {error_message}\n\n", status=503, mimetype='text/event-stream')

    generator = rag_chat.ask(
        question=question, 
        stream=True, 
        task_id=task_id, 
        bypass_assessment=bypass_assessment
    )
    return Response(generator, mimetype='text/event-stream')

@main.route('/api/write_report', methods=['GET'])
def write_report_from_blueprint():
    logging.info("--- API HIT: /api/write_report ---")
    if not rag_chat:
        logging.error("RAG service not initialized during write_report call.")
        error_message = json.dumps({"type": "error", "content": "Error: RAG service not initialized."})
        return Response(f"data: {error_message}\n\n", status=503, mimetype='text/event-stream')

    task_id = request.args.get('task_id')
    if not task_id:
        logging.warning("No task_id provided to /api/write_report.")
        error_message = json.dumps({"type": "error", "content": "Error: No task_id provided for report writing."})
        return Response(f"data: {error_message}\n\n", status=400, mimetype='text/event-stream')

    logging.info(f"Received request to write report for task_id: {task_id}")

    if task_id not in rag_chat.active_tasks:
        logging.error(f"Task ID {task_id} not found in active tasks.")
        error_message = json.dumps({"type": "error", "content": f"Error: Task ID {task_id} not found or has expired."})
        return Response(f"data: {error_message}\n\n", status=404, mimetype='text/event-stream')

    generator = rag_chat.stream_write_report(task_id=task_id)   
    return Response(generator, mimetype='text/event-stream')

@main.route('/api/cancel_task', methods=['POST'])
def cancel_task():
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

@main.route('/api/history', methods=['GET'])
def get_history():
    if not rag_chat: return jsonify({"error": "RAG service not initialized"}), 503
    history_list = rag_chat.get_history_list()
    return jsonify(history_list)

@main.route('/api/history/<int:entry_id>', methods=['GET'])
def get_history_entry(entry_id):
    if not rag_chat: return jsonify({"error": "RAG service not initialized"}), 503
    entry = rag_chat.get_history_entry(entry_id)
    if entry:
        return jsonify(entry)
    return jsonify({"error": "找不到該歷史紀錄"}), 404

@main.route('/api/history', methods=['POST'])
def add_history_entry():
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
    if not rag_chat: return jsonify({"error": "RAG service not initialized"}), 503
    if rag_chat.delete_history_entry(entry_id):
        return jsonify({"success": True})
    return jsonify({"error": "找不到該歷史紀錄或刪除失敗"}), 404

@main.route('/favicon.ico')
def favicon():
    return '', 204
