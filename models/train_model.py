import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_loader import load_data  # Charger les données depuis data_loader.py

# 🔹 Charger les données
df = load_data()

# 🔹 Traitement des données
# Convertir la colonne "Date" en format datetime et extraire des caractéristiques (ex : année, mois, jour)
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Encodage des variables catégorielles
label_encoder = LabelEncoder()

# Convertir les variables catégorielles en numériques
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Product Category'] = label_encoder.fit_transform(df['Product Category'])
df['Customer ID'] = label_encoder.fit_transform(df['Customer ID'])  # Encoder Customer ID également si nécessaire

# Sélectionner les colonnes features (X) et la cible (y)
X = df[["Transaction ID", "Year", "Month", "Day", "Customer ID", "Gender", "Age", "Product Category", "Quantity", "Price per Unit"]]
y = df["Total Amount"]

# 🔹 Normaliser les données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 🔹 Diviser les données en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Entraîner un modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# 🔹 Sauvegarder le modèle dans un fichier `model.pkl`
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modèle entraîné et sauvegardé avec succès dans models/model.pkl !")

