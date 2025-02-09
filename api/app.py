from flask import Flask
from flask_cors import CORS
from routes.home import home_bp
from routes.predict import ml_bp

app = Flask(__name__)
CORS(app)

# Enregistrer les routes
app.register_blueprint(home_bp)
app.register_blueprint(ml_bp, url_prefix="/ml")

if __name__ == "__main__":
    app.run(debug=True)
