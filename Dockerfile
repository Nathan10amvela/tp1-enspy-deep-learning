# Étape 1: Utiliser une image de base officielle Python
# On part d'une base légère qui contient déjà Python 3.9 installé.
FROM python:3.9-slim

# Étape 2: Définir le répertoire de travail dans le conteneur
# On crée un dossier /app à l'intérieur du conteneur et on s'y place.
WORKDIR /app

# Étape 3: Copier le fichier des dépendances et les installer
# On copie d'abord ce fichier seul pour profiter du cache de Docker.
# Si requirements.txt ne change pas, Docker n'exécutera pas l'étape d'installation à chaque build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Étape 4: Copier le reste de l'application
# On copie tous les autres fichiers (app.py, mnist_model.h5) dans le conteneur.
COPY . .

# Étape 5: Exposer le port de l'application
# On indique à Docker que l'application à l'intérieur du conteneur écoutera sur le port 5000.
EXPOSE 5000

# Étape 6: Commande pour démarrer l'application
# C'est la commande qui sera exécutée au lancement du conteneur.
CMD ["python", "app.py"]