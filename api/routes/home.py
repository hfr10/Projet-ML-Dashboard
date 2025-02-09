from flask import Blueprint, jsonify

home_bp = Blueprint("home", __name__)  # Définition du Blueprint

@home_bp.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API de Machine Learning est en ligne"})
