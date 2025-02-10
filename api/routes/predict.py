from flask import Blueprint, request, jsonify
import numpy as np
from models.load_model import model  # Importer le modèle chargé depuis models/load_model.py

ml_bp = Blueprint("ml", __name__)

@ml_bp.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer les données envoyées par le client dans la requête
        data = request.json["features"]  

        # Faire la prédiction avec le modèle
        prediction = model.predict([np.array(data)])

        # Retourner la prédiction sous forme de JSON
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        # En cas d'erreur, retourner un message d'erreur dans la réponse
        return jsonify({"error": str(e)}), 400

