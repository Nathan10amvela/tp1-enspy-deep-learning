import tensorflow as tf
from tensorflow import keras 
import numpy as np
import mlflow              ### MLFLOW ### : Importation de la bibliothèque MLflow
import mlflow.tensorflow   ### MLFLOW ### : Module spécifique pour TensorFlow/Keras


mlflow.tensorflow.autolog()  ### MLFLOW ### : Activation de l'autologging pour TensorFlow/Keras

# --- 2. Définition des paramètres ---
### MLFLOW ### : On centralise les hyperparamètres dans des variables.
### C'est une bonne pratique pour les logger et les modifier facilement.
EPOCHS = 10
BATCH_SIZE = 128
DROPOUT_RATE = 0.5 
L2_RATE = 0.001 

#Chargement du jeu de donnees MNIST
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

x_val = x_train_full[54000:]
y_val = y_train_full[54000:]
x_train = x_train_full[:54000]
y_train = y_train_full[:54000]

#Normalisation des donnees
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#Redimensionnement des images pour les reseaux fully-connected
x_train = x_train.reshape(54000, 784)
x_val = x_val.reshape(6000, 784)
x_test = x_test.reshape(10000, 784)

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(
            512, 
            activation='relu', 
            kernel_regularizer=keras.regularizers.l2(L2_RATE), 
            input_shape=(784,)
        ),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

optimizers = {
    'SGD_with_momentum': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'RMSprop': 'rmsprop',
    'Adam': 'adam'
}

mlflow.set_experiment("TP2 - Batch Normalization")

with mlflow.start_run(run_name="Model_With_BatchNorm"):
    

    model = keras.Sequential([
        keras.layers.Dense(
            512, 
            activation='relu', 
            kernel_regularizer=keras.regularizers.l2(L2_RATE), 
            input_shape=(784,)
        ),
        # On ajoute la Batch Normalization ici
        keras.layers.BatchNormalization(),
        
        keras.layers.Dropout(DROPOUT_RATE),
        
        keras.layers.Dense(10, activation='softmax')
    ])

    # --- 6. Compilation du modèle ---
    model.compile(
        optimizer='adam', # On garde le meilleur optimiseur
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- 7. Entraînement du modèle ---
    print("\n--- Démarrage de l'entraînement du modèle avec Batch Normalization ---")
    model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        verbose=1 # On remet verbose=1 pour bien voir la vitesse
    )

    # --- 8. Évaluation du modèle ---
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Précision finale sur le test set : {test_acc:.4f}")

    mlflow.log_metric("test_accuracy", test_acc)
    print("\nExpérience auto-enregistrée avec MLflow.")

#Sauvegarde du modele
model.save('mnist_model.h5')
print("Modele sauvegarde sous le nom 'mnist_model.h5'")