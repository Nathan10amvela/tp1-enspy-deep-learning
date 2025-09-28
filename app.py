from flask import Flask, request, jsonify  
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)


try:
    model = keras.models.load_model('mnist_model.h5')
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None


from flask import Flask, request, jsonify
import numpy as np
from PIL import Image

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': "Le modèle n'a pas pu être chargé"}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Aucune image fournie'}), 400

    file = request.files['file']


    image = Image.open(file).convert('L')

    # Redimensionner en 28x28
    image = image.resize((28, 28))

    # Convertir en numpy array
    image_data = np.array(image)

    
    image_data = 255 - image_data

    # Normaliser (les valeurs des pixels passent de 0-255 à 0.0-1.0)
    image_data = image_data.astype("float32") / 255.0

    # Aplatir l'image pour le modèle (de 28x28 à un vecteur de 784)
    image_data = image_data.reshape(1, 784)
    

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