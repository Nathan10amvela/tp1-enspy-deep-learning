import tensorflow as tf
from tensorflow import keras 
import numpy as np
import mlflow              ### MLFLOW ### : Importation de la bibliothèque MLflow
import mlflow.tensorflow   ### MLFLOW ### : Module spécifique pour TensorFlow/Keras

# --- 2. Définition des paramètres ---
### MLFLOW ### : On centralise les hyperparamètres dans des variables.
### C'est une bonne pratique pour les logger et les modifier facilement.
EPOCHS = 5
BATCH_SIZE = 128
DROPOUT_RATE = 0.2

#Chargement du jeu de donnees MNIST
((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()

#Normalisation des donnees
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#Redimensionnement des images pour les reseaux fully-connected
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Lancement de la session de suivi MLflow
with mlflow . start_run () :

    #Construction du modele
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    #Compilation du modele
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    #Entrainement du modele
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

    #Evaluation du modele
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Precision sur les donnees de test: {test_acc:.4f}')

    # --- 8. Enregistrement avec MLflow ---
    ### MLFLOW ### : C'est ici que l'on envoie les informations à MLflow.
    
    # a) Logger les paramètres
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dropout_rate", DROPOUT_RATE)
    
    # b) Logger les métriques
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_loss", test_loss)
    
    # c) Logger le modèle (l'artefact)
    # C'est plus puissant que model.save() car MLflow sauvegarde aussi
    # l'environnement (conda.yaml) et la signature du modèle.
    mlflow.keras.log_model(model, "mnist-model")

    print("\nExpérience, paramètres, métriques et modèle enregistrés avec MLflow.")

#Sauvegarde du modele
model.save('mnist_model.h5')
print("Modele sauvegarde sous le nom 'mnist_model.h5'")