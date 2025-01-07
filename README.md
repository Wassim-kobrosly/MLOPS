
 
-Model définit entreainé et enregistré avec Mlflow

-Site Web crée (voir https://github.com/Wassim-kobrosly/Flask-Web-App.git)

!! Prochaine étape CI/CD Gitlab 

!! Et enfin hébergement orender


MLOPS_Project/

├── src/                     # Scripts principaux du projet

│   ├── recommend.py         # Script principal pour l'entraînement et la prédiction selon les algo de factorisation et similaritè cosinus

│   ├── custom_model.py      # Exemple pour le prétraitement des données avec MLflow

│   ├── train.py             # Entraînement du modèle

│   └── watchfile.py         # Un watchdog qui serveille le moindre changement de interraction.csv et si oui execute recommand.py

├── mlruns/                  # Répertoire pour les logs MLflow (généré automatiquement)

├── Dockerfile               # Fichier Docker pour conteneurisation

├── requirements.txt         # Liste des dépendances Python
