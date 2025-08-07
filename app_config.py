"""
This file contains the central configuration for the application.
It now dynamically sets the database path based on the environment.
"""
import os
from pathlib import Path

# --- Core Paths ---
BASE_DIR = Path(__file__).resolve().parent

# --- Data Paths ---
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# --- Machine Learning Model Paths ---
ML_MODELS_DIR = BASE_DIR / "ml_models"
RUL_MODEL_DIR = ML_MODELS_DIR / "rul"
RUL_MODEL_PATH = RUL_MODEL_DIR / "rul_predictor.joblib"
RUL_SCALER_PATH = RUL_MODEL_DIR / "rul_scaler.pkl"
CLASSIFICATION_MODEL_DIR = ML_MODELS_DIR / "classification"
CLASSIFICATION_MODEL_PATH = CLASSIFICATION_MODEL_DIR / "fault_classifier.joblib"
CLASSIFICATION_PREPROCESSOR_PATH = CLASSIFICATION_MODEL_DIR / "classification_preprocessor.pkl"

# --- RAG and Knowledge Base Paths ---
# NEW: Use an environment variable for the persistent directory in production,
# but default to a local './db' folder for development.
DB_PATH = os.getenv("PERSIST_DIRECTORY", BASE_DIR / "db")

# --- Model-specific Hyperparameters ---
RUL_WINDOW_SIZE = 30
RUL_CAP = 125