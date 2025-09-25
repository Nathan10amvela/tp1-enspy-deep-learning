# --- 1. Importations ---
from flask import Flask, request, jsonify  # Flask pour créer l'API, request pour accéder aux données, jsonify pour formater la réponse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

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
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': "Le modèle n'a pas pu être chargé"}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Aucune image fournie'}), 400

    file = request.files['file']

    # --- ÉTAPE 1 (AJOUTÉE) : Conversion en niveaux de gris ---
    # Le modèle MNIST ne connaît que les niveaux de gris.
    # La méthode 'L' convertit l'image en luminance (niveaux de gris).
    image = Image.open(file).convert('L')

    # Redimensionner en 28x28
    image = image.resize((28, 28))

    # Convertir en numpy array
    image_data = np.array(image)

    # --- ÉTAPE 2 (AJOUTÉE) : Inversion des couleurs ---
    # Les images MNIST ont un chiffre blanc sur fond noir. 
    # Une image standard (chiffre noir sur fond blanc) a des valeurs de pixels élevées pour le fond
    # et basses pour le chiffre. On inverse cela.
    image_data = 255 - image_data

    # Normaliser (les valeurs des pixels passent de 0-255 à 0.0-1.0)
    image_data = image_data.astype("float32") / 255.0

    # Aplatir l'image pour le modèle (de 28x28 à un vecteur de 784)
    image_data = image_data.reshape(1, 784)
    
    # Le code suivant pour l'affichage peut être utile pour le débogage
    # plt.imshow(image_data.reshape(28, 28), cmap="gray")
    # plt.show()

    # Faire la prédiction
    prediction_probs = model.predict(image_data)
    predicted_class = np.argmax(prediction_probs, axis=1)[0]

    return jsonify({
        'prediction': int(predicted_class),
        'probabilities': prediction_probs.tolist()
    })

if __name__ == '__main__':
    # Assurez-vous que le modèle est chargé avant de lancer l'app
    app.run(host='0.0.0.0', port=5000)