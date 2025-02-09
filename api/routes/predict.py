from flask import Blueprint, request, jsonify
import numpy as np
from models.load_model import model  # Importer le modèle chargé

ml_bp = Blueprint("ml", __name__)

@ml_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]  
        prediction = model.predict([np.array(data)])
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
