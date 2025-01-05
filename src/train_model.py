import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.sklearn

# Charger les fichiers CSV
products = pd.read_csv('../../test/data/products.csv')
interactions = pd.read_csv('../../test/data/interactions.csv')

# Ajouter une colonne 'id_produit' si elle n'existe pas
if 'id_produit' not in products.columns:
    products['id_produit'] = range(1, len(products) + 1)

# Filtrer les interactions pour inclure uniquement les produits existants
interactions = interactions[interactions['product_id'].isin(products['id_produit'])]

# Ajouter la colonne 'interaction_id' si elle n'existe pas
if 'interaction_id' not in interactions.columns:
    interactions['interaction_id'] = range(1, len(interactions) + 1)

# Prétraitement des données produits
products['text_features'] = products['main_category'] + " " + products['sub_category'] + " " + products['name']
tfidf = TfidfVectorizer()
product_vectors = tfidf.fit_transform(products['text_features'])

# Fonction d'entraînement du modèle
def train_model():
    with mlflow.start_run():
        # Enregistrer le modèle
        mlflow.sklearn.log_model(tfidf, "tfidf_vectorizer")
        mlflow.sklearn.log_model(product_vectors, "product_vectors")

        # Enregistrer les paramètres et les métriques
        mlflow.log_param("num_products", len(products))
        mlflow.log_metric("tfidf_vocab_size", len(tfidf.vocabulary_))

    print("Modèle entraîné et enregistré avec MLflow.")

if __name__ == '__main__':
    train_model()

