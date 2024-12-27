# model_training.py

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def train_model(data_dir, model_dir):
    """
    Entraîner le modèle de recommandation basé sur TF-IDF et sauvegarder les vecteurs.

    Args:
        data_dir (str): Chemin du dossier contenant les datasets prétraités.
        model_dir (str): Chemin pour sauvegarder les fichiers du modèle.
    """
    # Charger les produits prétraités
    products_file = os.path.join(data_dir, 'processed_products.csv')
    if not os.path.exists(products_file):
        raise FileNotFoundError(f"Fichier manquant: {products_file}")

    products = pd.read_csv(products_file)

    # Créer une colonne de caractéristiques textuelles
    products['text_features'] = products['main_category'] + " " + products['sub_category'] + " " + products['name']

    # Vectorisation TF-IDF
    tfidf = TfidfVectorizer()
    product_vectors = tfidf.fit_transform(products['text_features'])

    # Sauvegarder le vecteur TF-IDF et les produits
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf, f)

    with open(os.path.join(model_dir, 'product_vectors.pkl'), 'wb') as f:
        pickle.dump(product_vectors, f)

    products.to_csv(os.path.join(model_dir, 'products.csv'), index=False)

    print("Modèle entraîné et sauvegardé.")

if __name__ == "__main__":
    data_dir = "~/desktop/mlops/data"
    model_dir = "~/desktop/mlops/model"
    train_model(data_dir, model_dir)
