import pandas as pd

def load_data():
    """Charge et nettoie les données depuis dataset.csv"""
    df = pd.read_csv("data/dataset.csv")  
    df.dropna(inplace=True)  # Supprime les valeurs manquantes
    return df
