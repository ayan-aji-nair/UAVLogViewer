import pickle
import pandas as pd
import os
import gzip
from pathlib import Path

def save_dataframes_to_pickle(dataframes_dict, filepath, compress=True):
    """
    Save a dictionary of DataFrames to a pickle file.
    
    Args:
        dataframes_dict: Dictionary mapping codes to DataFrames
        filepath: Path to save the pickle file
        compress: Whether to compress the file with gzip
    """
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    if compress:
        filepath = filepath + '.gz'
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(dataframes_dict, f)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(dataframes_dict, f)
    
    print(f"Saved {len(dataframes_dict)} DataFrames to {filepath}")

def load_dataframes_from_pickle(filepath):
    """
    Load a dictionary of DataFrames from a pickle file.
    
    Args:
        filepath: Path to the pickle file (with or without .gz extension)
        
    Returns:
        Dictionary mapping codes to DataFrames
    """
    # Check if compressed file exists
    if os.path.exists(filepath + '.gz'):
        filepath = filepath + '.gz'
        with gzip.open(filepath, 'rb') as f:
            dataframes_dict = pickle.load(f)
    elif os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            dataframes_dict = pickle.load(f)
    else:
        print(f"File {filepath} does not exist")
        return {}
    
    print(f"Loaded {len(dataframes_dict)} DataFrames from {filepath}")
    return dataframes_dict

# Example usage:
# dataframes = {"ATT": df1, "GPS": df2, "IMU": df3}
# save_dataframes_to_pickle(dataframes, "backend/dataframes/log_data.pkl", compress=True)
# loaded_dataframes = load_dataframes_from_pickle("backend/dataframes/log_data.pkl") 