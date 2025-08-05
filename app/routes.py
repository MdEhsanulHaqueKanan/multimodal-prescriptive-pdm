from flask import request, jsonify, render_template
from app import app
from app.utils import (
    get_rul_prediction, 
    get_fault_prediction, 
    simulate_drift_detection, 
    get_dashboard_data,
    get_rag_chain_instance
)
import pandas as pd
import app_config as config
import logging

# Load Data on App Start (only data, not models)
try:
    rul_df_processed = pd.read_csv(config.PROCESSED_DATA_DIR / "rul_processed_data.csv")
except FileNotFoundError:
    rul_df_processed = None
    logging.warning("RUL processed data not found. Dashboard for RUL assets will be affected.")

# --- Main Application Routes ---

@app.route('/healthz')
def healthz():
    """Simple health check endpoint for Render."""
    return "OK", 200

@app.route('/')
@app.route('/dashboard')
def dashboard():
    dashboard_data = get_dashboard_data(rul_df_processed)
    return render_template('dashboard.html', title='Dashboard', assets=dashboard_data.get('assets', []), topics=dashboard_data.get('topics', []))

@app.route('/asset/<asset_id>')
def asset_detail(asset_id):
    if 'Turbofan' in asset_id:
        asset_type = 'rul'
        try: unit_number = int(asset_id.split('#')[-1])
        except (ValueError, IndexError): return "Invalid Turbofan ID format", 404
        if rul_df_processed is not None:
            asset_history_df = rul_df_processed[rul_df_processed['unit_number'] == unit_number].copy()
            historical_data_json = asset_history_df.to_dict(orient='list')
        else: historical_data_json = {}
    elif 'Machine' in asset_id or 'Conveyor' in asset_id:
        asset_type, historical_data_json = 'classification', {}
    else: return "Unknown Asset Type", 404
    return render_template('asset_detail.html', title=f"Asset: {asset_id}", asset_id=asset_id, asset_type=asset_type, historical_data=historical_data_json)

@app.route('/model-monitor')
def model_monitor():
    monitoring_data = simulate_drift_detection()
    return render_template('model_monitor.html', title="Model Monitor", monitoring_data=monitoring_data)

@app.route('/about')
def about():
    return render_template('about.html', title="About")

# --- API Endpoints ---

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    prediction_type, payload = data.get('type'), data.get('data')
    df = pd.DataFrame(payload)
    if prediction_type == 'rul':
        result = get_rul_prediction(df, with_shap=True)
    elif prediction_type == 'classification':
        result = get_fault_prediction(df)
    else: return jsonify({"error": "Invalid prediction type"}), 400
    if "error" in result: return jsonify(result), 500
    return jsonify(result)

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get("question")
    rag_chain = get_rag_chain_instance()
    if rag_chain is None:
        return jsonify({"error": "RAG chain is not available."}), 500
    try:
        answer = rag_chain.invoke(question)
        return jsonify({"answer": answer})
    except Exception as e:
        logging.error(f"Error in RAG chain: {e}", exc_info=True)
        return jsonify({"error": "Failed to process question."}), 500

@app.route('/api/feedback', methods=['POST'])
def handle_feedback():
    data = request.get_json()
    logging.info(f"--- FEEDBACK RECEIVED ---: {data}")
    return jsonify({"status": "success"}), 200