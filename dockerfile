# Utiliser une image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail dans le container
WORKDIR /app

# Copier le fichier requirements.txt dans le container
COPY requirements.txt /app/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source dans le container
COPY . /app/

# Exposer le port de l'API
EXPOSE 5000

# Commande pour démarrer l'API Flask
CMD ["python", "app.py"]
