# Utiliser une image Python officielle comme base
FROM python:3.8-slim

# Installer des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Mettre à jour pip
RUN pip install --upgrade pip

# Définir le répertoire de travail
WORKDIR /app

# Copier les dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY ./src .

# Exposer le port
EXPOSE 5000

# Commande de démarrage
CMD ["python", "api.py"]
