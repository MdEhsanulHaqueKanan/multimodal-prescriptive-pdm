import pandas as pd
import numpy as np
import sys
from pathlib import Path

# --- Add Project Root to the Python Path ---
try:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    project_root = str(Path(__file__).resolve().parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

import app_config as config # Use our new, renamed config file

def generate_data(n_samples, concept, random_state=42):
    """Generates synthetic data for a given concept."""
    np.random.seed(random_state)
    X1 = np.random.rand(n_samples) * 10
    X2 = np.random.randn(n_samples) * 5
    
    if concept == 'A':
        y = (2 * X1 + X2/2 + np.random.randn(n_samples)) > 10
    else: # Concept B
        y = (2 * (10 - X1) + X2/2 + np.random.randn(n_samples)) > 10
        
    df = pd.DataFrame({'feature1': X1, 'feature2': X2, 'target': y.astype(int)})
    return df

def generate_drift_dataset():
    """Generates and saves a dataset with an abrupt concept drift."""
    print("--- Generating Synthetic Drift Dataset ---")
    
    df_concept_a = generate_data(500, 'A', random_state=42)
    df_concept_b = generate_data(500, 'B', random_state=43)
    
    drift_df = pd.concat([df_concept_a, df_concept_b], ignore_index=True)
    
    # Use the new config path
    # Ensure the 'processed' directory exists
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.PROCESSED_DATA_DIR / "drift_simulation_data.csv"
    
    drift_df.to_csv(output_path, index=False)
    
    print(f"Successfully generated and saved drift dataset with {len(drift_df)} samples.")
    print(f"Dataset saved to: {output_path}")

if __name__ == '__main__':
    generate_drift_dataset()