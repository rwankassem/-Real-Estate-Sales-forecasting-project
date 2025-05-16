# src/data_loader.py
import pandas as pd
def load_data(path):
    return pd.read_csv(path)