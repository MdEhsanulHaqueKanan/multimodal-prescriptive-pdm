import os
import logging
from flask import Flask
from data_ingestion.ingest_text import run_ingestion_if_needed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Run data ingestion on startup if the DB doesn't exist.
# This is crucial for cloud deployment.
run_ingestion_if_needed()

# Initialize the Flask app
app = Flask(__name__)

logging.info("Flask app initialized. Models will be loaded on first use.")

# Import the routes after the app is created to avoid circular imports
from app import routes