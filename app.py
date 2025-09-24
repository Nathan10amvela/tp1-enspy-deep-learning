# --- 1. Importations ---
from flask import Flask, request, jsonify  # Flask pour créer l'API, request pour accéder aux données, jsonify pour formater la réponse
import tensorflow as tf
from tensorflow import keras
import numpy as np

# --- 2. Initialisation de l'application Flask ---
app = Flask(__name__)

# --- 3. Chargement du modèle Keras ---
# Le modèle est chargé une seule fois au démarrage du serveur pour être efficace.
# Assurez-vous que le fichier 'mnist_model.h5' est dans le même dossier.
try:
    model = keras.models.load_model('mnist_model.h5')
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None

# --- 4. Définition de l'endpoint de prédiction ---
# @app.route définit une URL. Ici, c'est l'adresse '/predict'.
# methods=['POST'] signifie que cette URL n'accepte que les requêtes de type POST (envoi de données).
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Le modèle n''a pas pu être chargé'}), 500

    # a) Récupérer les données JSON de la requête entrante
    data = request.json
    
    # b) Vérification simple des données
    if 'image' not in data:
        return jsonify({'error': 'Aucune image fournie'}), 400 # Bad Request

    # c) Prétraitement de l'image (exactement comme dans le script d'entraînement !)
    image_data = np.array(data['image'])
    
    # Le modèle attend un format (batch_size, 784), donc (1, 784) pour une seule image.
    image_data = image_data.reshape(1, 784)
    image_data = image_data.astype("float32") / 255.0

    # d) Faire la prédiction
    prediction_probs = model.predict(image_data)
    
    # e) Formater la réponse
    predicted_class = np.argmax(prediction_probs, axis=1)[0]
    
    # On renvoie une réponse JSON propre.
    return jsonify({
        'prediction': int(predicted_class), # Le chiffre prédit
        'probabilities': prediction_probs.tolist() # La liste des probabilités pour chaque chiffre
    })

# --- 5. Lancement du serveur ---
# Cette partie s'exécute seulement si on lance le script directement (python app.py).
if __name__ == '__main__':
    # host='0.0.0.0' rend le serveur accessible depuis l'extérieur du conteneur Docker.
    app.run(host='0.0.0.0', port=5000)