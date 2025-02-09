import pickle

# Charger le modèle
def load_trained_model():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_trained_model()
