"""
This file contains the central configuration for the application.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ML_MODELS_DIR = BASE_DIR / "ml_models"
RUL_MODEL_DIR = ML_MODELS_DIR / "rul"
RUL_MODEL_PATH = RUL_MODEL_DIR / "rul_predictor.joblib"
RUL_SCALER_PATH = RUL_MODEL_DIR / "rul_scaler.pkl"
CLASSIFICATION_MODEL_DIR = ML_MODELS_DIR / "classification"
CLASSIFICATION_MODEL_PATH = CLASSIFICATION_MODEL_DIR / "fault_classifier.joblib"
CLASSIFICATION_PREPROCESSOR_PATH = CLASSIFICATION_MODEL_DIR / "classification_preprocessor.pkl"
DB_PATH = BASE_DIR / "db"
RUL_WINDOW_SIZE = 30
RUL_CAP = 125