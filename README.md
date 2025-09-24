# Travaux Pratiques: De la conception au déploiement de modèles de Deep Learning

Ce dépôt contient le code source pour le projet de Deep Learning Engineering réalisé dans le cadre du cours du Département Génie Informatique de l'ENSPY.

Le projet couvre le cycle de vie complet d'un modèle de Machine Learning, de l'entraînement d'un réseau de neurones à son déploiement en tant que service conteneurisé, en passant par le suivi des expérimentations.

## Table des Matières
1. [Objectifs Pédagogiques](#objectifs-pédagogiques)
2. [Technologies Utilisées](#technologies-utilisées)
3. [Structure du Projet](#structure-du-projet)
4. [Installation](#installation)
5. [Utilisation](#utilisation)
   - [Partie 1 : Entraînement du Modèle](#partie-1--entraînement-du-modèle)
   - [Partie 2 : Suivi avec MLflow](#partie-2--suivi-avec-mlflow)
   - [Partie 3 : API Web avec Flask](#partie-3--api-web-avec-flask)
   - [Partie 4 : Conteneurisation avec Docker](#partie-4--conteneurisation-avec-docker)
6. [Auteurs](#auteurs)

## Objectifs Pédagogiques
- Comprendre les concepts fondamentaux de l'apprentissage machine et profond.
- Maîtriser les étapes du cycle de vie d'un modèle de Deep Learning.
- Apprendre à utiliser **Git** et **GitHub** pour la collaboration et le versionnement.
- Découvrir et utiliser **MLflow** pour le suivi des expérimentations.
- Savoir empaqueter un modèle dans une application web avec **Flask** et le conteneuriser avec **Docker**.

## Technologies Utilisées
- **Framework de Deep Learning :** TensorFlow, Keras
- **Bibliothèques Python :** NumPy, Scikit-learn
- **MLOps :** MLflow, Docker
- **Serveur Web :** Flask
- **Gestion de l'environnement :** Conda

## Structure du Projet
.
├── mnist_model.h5 # Modèle Keras entraîné et sauvegardé
├── train_model.py # Script pour l'entraînement du modèle (intégrant MLflow)
├── app.py # Script de l'API Flask pour servir le modèle
├── Dockerfile # Fichier de configuration pour construire l'image Docker
├── requirements.txt # Fichier listant les dépendances Python du projet
└── README.md # Ce fichier


## Installation

Suivez ces étapes pour configurer l'environnement de développement local.

### Prérequis
- [Git](https://git-scm.com/)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou Anaconda

### Étapes
1.  **Clonez le dépôt :**
    ```bash
    git clone https://github.com/votre-nom-utilisateur/votre-repo.git
    cd votre-repo
    ```

2.  **Créez et activez l'environnement Conda :**
    ```bash
    conda create --name tp-deep-learning python=3.9
    conda activate tp-deep-learning
    ```

3.  **Installez les dépendances :**
    Un fichier `requirements.txt` sera créé pour lister toutes les dépendances. Pour l'instant, vous pouvez les installer manuellement :
    ```bash
    pip install tensorflow numpy mlflow flask
    ```

## Utilisation

### Partie 1 : Entraînement du Modèle
Pour entraîner le réseau de neurones sur le jeu de données MNIST et sauvegarder le modèle :
```bash
python train_model.py
```
Cette commande exécutera le script, affichera la précision du modèle et créera le fichier mnist_model.h5.

### Partie 2 : Suivi avec MLflow
Le script train_model.py est configuré pour enregistrer les paramètres et les métriques de chaque exécution avec MLflow.

1. Lancez une exécution trackée (la commande est la même que ci-dessus) :

```bash
python train_model.py
```

2. Visualisez les résultats en lançant l'interface utilisateur de MLflow. Dans le même dossier, exécutez :

```bash
mlflow ui
```

3. Ouvrez votre navigateur et allez à l'adresse http://127.0.0.1:5000 pour comparer les différentes exécutions.

### Partie 3 : API Web avec Flask
Pour servir le modèle entraîné via une API REST.

1. Lancez le serveur Flask :
```bash
    python app.py
```
2. Le serveur sera accessible sur http://127.0.0.1:5000. Vous pouvez envoyer des requêtes POST à l'endpoint /predict avec un outil comme curl ou Postman.

### Partie 4 : Conteneurisation avec Docker
Pour empaqueter l'application et ses dépendances dans une image Docker.

1. Construisez l'image Docker :
```bash
docker build -t mnist-api .
```

2. Lancez un conteneur à partir de l'image :
``` bash
docker run -p 5000:5000 mnist-api
```

L'API sera alors accessible de la même manière, via http://127.0.0.1:5000.

### Auteurs
Ce projet est basé sur les travaux pratiques conçus par :

- Louis Fippo Fitime

- Claude Tinku

- Kerolle Sonfack

Département Génie Informatique, ENSPY.