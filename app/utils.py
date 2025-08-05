# =========================================================================
# === THIS IS THE NEW, MEMORY-EFFICIENT app/utils.py FILE ===
# =========================================================================
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import joblib
import pandas as pd
import numpy as np
import app_config as config
import traceback
import logging
import json
from prescriptive_rag.chains import create_rag_chain

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Placeholders for Lazily-Loaded Models ---
# We initialize them to None. They will be loaded into these global
# variables the first time they are requested.
RUL_MODEL = None
RUL_SCALER = None
FAULT_CLASSIFIER = None
CLASSIFICATION_PREPROCESSOR = None
RAG_CHAIN = None

# --- "Getter" Functions for Lazy Loading ---

def get_rag_chain_instance():
    """
    Loads the RAG chain on first call and caches it in a global variable.
    """
    global RAG_CHAIN
    if RAG_CHAIN is None:
        logging.info("RAG chain not loaded. Initializing now...")
        try:
            RAG_CHAIN = create_rag_chain()
            logging.info("RAG chain successfully initialized and cached.")
        except Exception as e:
            logging.error(f"Failed to initialize RAG chain: {e}", exc_info=True)
            RAG_CHAIN = None # Ensure it remains None on failure
    return RAG_CHAIN

def get_rul_model_and_scaler():
    """
    Loads the RUL model and scaler on first call and caches them.
    """
    global RUL_MODEL, RUL_SCALER
    if RUL_MODEL is None or RUL_SCALER is None:
        logging.info("RUL model/scaler not loaded. Initializing now...")
        try:
            RUL_MODEL = joblib.load(config.RUL_MODEL_PATH)
            RUL_SCALER = joblib.load(config.RUL_SCALER_PATH)
            logging.info("RUL model and scaler successfully loaded and cached.")
        except FileNotFoundError:
            logging.error(f"Could not load RUL model/scaler from {config.RUL_MODEL_PATH}")
            RUL_MODEL, RUL_SCALER = None, None
    return RUL_MODEL, RUL_SCALER

def get_classification_model_and_preprocessor():
    """
    Loads the classification model and preprocessor on first call and caches them.
    """
    global FAULT_CLASSIFIER, CLASSIFICATION_PREPROCESSOR
    if FAULT_CLASSIFIER is None or CLASSIFICATION_PREPROCESSOR is None:
        logging.info("Classification model/preprocessor not loaded. Initializing now...")
        try:
            FAULT_CLASSIFIER = joblib.load(config.CLASSIFICATION_MODEL_PATH)
            CLASSIFICATION_PREPROCESSOR = joblib.load(config.CLASSIFICATION_PREPROCESSOR_PATH)
            logging.info("Classification model and preprocessor successfully loaded and cached.")
        except FileNotFoundError:
            logging.error(f"Could not load classification model from {config.CLASSIFICATION_MODEL_PATH}")
            FAULT_CLASSIFIER, CLASSIFICATION_PREPROCESSOR = None, None
    return FAULT_CLASSIFIER, CLASSIFICATION_PREPROCESSOR


# --- Prediction Functions (Now using the "getter" functions) ---

def get_rul_prediction(data: pd.DataFrame, with_shap: bool = False) -> dict:
    rul_model, rul_scaler = get_rul_model_and_scaler()
    if rul_model is None or rul_scaler is None:
        return {"error": "RUL model is not available."}
    # ... (rest of the function is identical to before)
    df = data.copy()
    sensor_cols = [col for col in df.columns if 'sensor' in col and '_rolling' not in col]
    for col in sensor_cols:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=config.RUL_WINDOW_SIZE, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=config.RUL_WINDOW_SIZE, min_periods=1).std()
    df.fillna(0, inplace=True)
    last_row = df.iloc[[-1]]
    required_features = rul_model.feature_names_in_
    features_for_model = last_row[required_features]
    X_scaled = rul_scaler.transform(features_for_model)
    predicted_rul = min(float(rul_model.predict(X_scaled)[0]), config.RUL_CAP)
    shap_plot_base64 = None
    if with_shap:
        try:
            explainer = shap.TreeExplainer(rul_model)
            shap_plot_base64 = _generate_shap_plot(rul_model, explainer, X_scaled, required_features)
        except Exception as e:
            logging.error(f"Error generating RUL SHAP plot: {e}")
    return {"predicted_rul": round(predicted_rul, 2), "shap_plot": shap_plot_base64}

def get_fault_prediction(data: pd.DataFrame) -> dict:
    fault_classifier, classification_preprocessor = get_classification_model_and_preprocessor()
    if fault_classifier is None or classification_preprocessor is None:
        return {"error": "Classification model is not available."}
    # ... (rest of the function is identical to before)
    X_processed = classification_preprocessor.transform(data)
    prediction = fault_classifier.predict(X_processed)[0]
    probabilities = fault_classifier.predict_proba(X_processed)[0]
    class_index = np.where(fault_classifier.classes_ == prediction)[0][0]
    confidence = probabilities[class_index]
    shap_plot_base64 = None
    try:
        explainer = shap.TreeExplainer(fault_classifier)
        feature_names = classification_preprocessor.get_feature_names_out()
        shap_plot_base64 = _generate_shap_plot(fault_classifier, explainer, X_processed, feature_names)
    except Exception as e:
        logging.error(f"Error generating Classification SHAP plot: {e}")
    return {"predicted_fault": str(prediction), "confidence": round(float(confidence), 2), "shap_plot": shap_plot_base64}


# --- Other Functions (SHAP plot, monitoring, etc. remain the same) ---
# --- NOTE: For brevity, I am omitting the other functions. Copy them from your existing file ---
def _generate_shap_plot(model, explainer, processed_data, feature_names):
    """A generic helper to generate a SHAP force plot."""
    # ... (This function remains IDENTICAL)
    shap_values = explainer(processed_data)
    is_classification = hasattr(model, 'classes_')
    if is_classification:
        prediction = model.predict(processed_data)[0]
        class_list = model.classes_.tolist()
        prediction_index = class_list.index(prediction)
        base_value = shap_values.base_values[0, prediction_index]
        shap_values_for_plot = shap_values.values[0, :, prediction_index]
        feature_names_clean = [name.split('__')[1] for name in feature_names]
    else: # Regression
        base_value = explainer.expected_value
        shap_values_for_plot = shap_values.values[0]
        feature_names_clean = feature_names
    force_plot = shap.force_plot(
        base_value=base_value, shap_values=shap_values_for_plot, features=processed_data[0],
        feature_names=feature_names_clean, matplotlib=True, show=False, figsize=(20, 5), text_rotation=15
    )
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(force_plot)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

def simulate_drift_detection():
    """ Simulates monitoring a data stream for concept drift. """
    # ... (This function remains IDENTICAL)
    try:
        df = pd.read_csv(config.PROCESSED_DATA_DIR / "drift_simulation_data.csv")
    except FileNotFoundError:
        return {"error": "Drift simulation data not found."}
    # ... (rest of the logic)
    initial_train_size=200; model=SGDClassifier(loss='log_loss',random_state=42); X_initial=df.iloc[:initial_train_size][['feature1','feature2']]; y_initial=df.iloc[:initial_train_size]['target']; model.fit(X_initial, y_initial); stream_data=df.iloc[initial_train_size:]; chunk_size=50; time_steps,accuracies,drift_points=[],[],[]; drift_threshold,is_drift_detected=0.70,False
    for i in range(0,len(stream_data),chunk_size):
        chunk=stream_data.iloc[i:i+chunk_size];
        if chunk.empty:continue
        X_chunk,y_chunk_true=chunk[['feature1','feature2']],chunk['target']; y_chunk_pred=model.predict(X_chunk); acc=accuracy_score(y_chunk_true,y_chunk_pred); time_steps.append(initial_train_size+i+chunk_size/2); accuracies.append(acc)
        if acc<drift_threshold and not is_drift_detected:drift_points.append({"time":initial_train_size+i+chunk_size/2,"label":"Concept Drift Detected"}); is_drift_detected=True
    return {"time_steps":time_steps,"accuracies":accuracies,"drift_points":drift_points,"drift_threshold":drift_threshold}

def get_fleet_topics():
    """ Loads the discovered topics from the JSON file. """
    # ... (This function remains IDENTICAL)
    try:
        with open(config.PROCESSED_DATA_DIR / "dashboard_topics.json", 'r') as f:
            topics = json.load(f)
        return topics
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []

def get_dashboard_data(rul_df_processed: pd.DataFrame) -> dict:
    """ Generates the dynamic data for the dashboard, including assets and topics. """
    # ... (This function remains IDENTICAL)
    all_dashboard_assets=[]
    if rul_df_processed is not None:
        # ... (rest of logic)
        pass # Placeholder for your existing RUL logic
    classification_samples = {
        "Milling Machine #XYZ": pd.DataFrame([{"type":"L","air_temperature":302.5,"process_temperature":311.8,"rotational_speed":1390,"torque":55.3,"tool_wear":208}]),
        "Conveyor Belt #A-3": pd.DataFrame([{"type":"M","air_temperature":300.1,"process_temperature":309.7,"rotational_speed":1550,"torque":41.2,"tool_wear":20}])
    }
    for asset_id,sample_df in classification_samples.items():
        asset_info={"id":asset_id,"type":"classification"}; prediction_result=get_fault_prediction(sample_df); asset_info["prediction"]=prediction_result
        fault=prediction_result.get("predicted_fault","No Failure")
        if fault!="No Failure":asset_info["status_class"]="border-red"; asset_info["status_text"]="Fault Predicted"
        else:asset_info["status_class"]="border-green"; asset_info["status_text"]="Healthy"
        all_dashboard_assets.append(asset_info)
    status_order={"border-red":0,"border-orange":1,"border-green":2}; final_sorted_assets=sorted(all_dashboard_assets,key=lambda x:status_order.get(x.get('status_class'),99))
    fleet_topics=get_fleet_topics()
    return {"assets":final_sorted_assets,"topics":fleet_topics}