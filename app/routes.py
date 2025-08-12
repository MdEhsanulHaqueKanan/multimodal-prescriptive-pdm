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

# --- Main Application Routes ---

@app.route('/healthz')
def healthz():
    """Simple health check endpoint for cloud platforms."""
    return "OK", 200

@app.route('/')
@app.route('/dashboard')
def dashboard():
    """
    Renders the main dashboard page.
    It now loads the RUL data on-demand for robustness.
    """
    # Load the RUL data right when we need it
    try:
        rul_df = pd.read_csv(config.PROCESSED_DATA_DIR / "rul_processed_data.csv")
    except FileNotFoundError:
        rul_df = None
        logging.warning("RUL processed data not found. Dashboard for RUL assets will be affected.")
    
    # Pass the newly loaded dataframe to the utility function
    dashboard_data = get_dashboard_data(rul_df)
    
    return render_template(
        'dashboard.html', 
        title='Dashboard', 
        assets=dashboard_data.get('assets', []), 
        topics=dashboard_data.get('topics', [])
    )

@app.route('/asset/<asset_id>')
def asset_detail(asset_id):
    """
    Renders the detail page for a specific asset.
    It now loads the RUL data on-demand for robustness.
    """
    if 'Turbofan' in asset_id:
        asset_type = 'rul'
        historical_data_json = {} # Default to empty
        try: 
            unit_number = int(asset_id.split('#')[-1])
            # Load the RUL data right when we need it for a specific asset
            rul_df = pd.read_csv(config.PROCESSED_DATA_DIR / "rul_processed_data.csv")
            asset_history_df = rul_df[rul_df['unit_number'] == unit_number].copy()
            if not asset_history_df.empty:
                historical_data_json = asset_history_df.to_dict(orient='list')
        except (ValueError, IndexError): 
            return "Invalid Turbofan ID format", 404
        except FileNotFoundError:
             logging.warning("RUL processed data file not found when trying to load asset detail.")
        
    elif 'Machine' in asset_id or 'Conveyor' in asset_id:
        asset_type, historical_data_json = 'classification', {}
    else: 
        return "Unknown Asset Type", 404
        
    return render_template(
        'asset_detail.html', 
        title=f"Asset: {asset_id}", 
        asset_id=asset_id, 
        asset_type=asset_type, 
        historical_data=historical_data_json
    )

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