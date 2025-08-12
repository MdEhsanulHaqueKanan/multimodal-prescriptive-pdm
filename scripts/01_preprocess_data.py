import pandas as pd
import logging
import sys
from pathlib import Path

# --- Add Project Root to the Python Path ---
# This is a robust way to allow this script to import modules from the project root,
# like our app_config.py file.
try:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.append(project_root)
except IndexError:
    # Handle the case where the script is not in a subdirectory
    project_root = str(Path(__file__).resolve().parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

import app_config as config  # Now we can import our new config file

# --- Setup Professional Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Define Constants based on the old config ---
# These were hard-coded in your old config, so we define them here.
RUL_COLS_TO_DROP = [
    'op_setting_3', 'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
    'sensor_16', 'sensor_18', 'sensor_19'
]
CLASSIFICATION_COLS_TO_DROP = ['udi', 'product_id']

def _calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the Remaining Useful Life for each engine unit."""
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycles']
    df = pd.merge(df, max_cycles, on='unit_number', how='left')
    df['RUL'] = df['max_cycles'] - df['time_in_cycles']
    df.drop(columns=['max_cycles'], inplace=True)
    return df

def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds rolling mean and standard deviation features for sensor readings."""
    sensor_cols = [col for col in df.columns if col.startswith('sensor')]
    for col in sensor_cols:
        df[f'{col}_rolling_mean'] = df.groupby('unit_number')[col].rolling(
            window=config.RUL_WINDOW_SIZE, min_periods=1
        ).mean().reset_index(level=0, drop=True)
        df[f'{col}_rolling_std'] = df.groupby('unit_number')[col].rolling(
            window=config.RUL_WINDOW_SIZE, min_periods=1
        ).std().reset_index(level=0, drop=True)
    
    df.fillna(0, inplace=True)
    return df

def preprocess_rul_data():
    """Processes the raw NASA Turbofan dataset (FD001)."""
    logging.info("--- Starting RUL Data Preprocessing ---")

    column_names = [
        'unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3',
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6',
        'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12',
        'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18',
        'sensor_19', 'sensor_20', 'sensor_21'
    ]

    # Use the new config path
    rul_input_file = config.RAW_DATA_DIR / "c-mapss" / "train_FD001.txt"

    try:
        df = pd.read_csv(rul_input_file, sep=r'\s+', header=None, names=column_names)
        logging.info(f"Loaded raw RUL data with shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Raw RUL data not found at {rul_input_file}. Aborting.")
        return

    df = _calculate_rul(df)
    df.drop(columns=RUL_COLS_TO_DROP, inplace=True)
    logging.info(f"Dropped useless columns. Shape is now: {df.shape}")
    
    df = _add_rolling_features(df)
    logging.info(f"Added rolling features. Final shape: {df.shape}")
    
    df['RUL'] = df['RUL'].clip(upper=config.RUL_CAP)
    logging.info(f"Capped RUL at {config.RUL_CAP} cycles.")

    # Use the new config path
    output_path = config.PROCESSED_DATA_DIR / "rul_processed_data.csv"
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved processed RUL data to {output_path}")

def preprocess_classification_data():
    """Processes the raw Kaggle Predictive Maintenance dataset."""
    logging.info("--- Starting Classification Data Preprocessing ---")
    
    # Use the new config path
    classification_input_file = config.RAW_DATA_DIR / "ai4i2020.csv"

    try:
        df = pd.read_csv(classification_input_file)
        logging.info(f"Loaded raw classification data with shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Raw classification data not found at {classification_input_file}. Aborting.")
        return

    new_columns = [col.replace('[K]', '').replace('[rpm]', '').replace('[Nm]', '').replace('[min]', '').strip().replace(' ', '_').lower() for col in df.columns]
    df.columns = new_columns
    df = df.drop(columns=CLASSIFICATION_COLS_TO_DROP)
    logging.info(f"Cleaned column names and dropped ID columns. Final shape: {df.shape}")
    
    # Use the new config path
    output_path = config.PROCESSED_DATA_DIR / "classification_processed_data.csv"
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Successfully saved processed classification data to {output_path}")

def main():
    """Main function to run both preprocessing steps."""
    preprocess_rul_data()
    preprocess_classification_data()
    logging.info("--- All preprocessing is complete! ---")

if __name__ == '__main__':
    main()