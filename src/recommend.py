# recommend.py

import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def load_model(model_dir):
    """
    Charger les fichiers du modèle préentraîné.

    Args:
        model_dir (str): Chemin du dossier contenant les fichiers du modèle.

    Returns:
        tuple: Le vectoriseur TF-IDF, les vecteurs produits et le DataFrame produits.
    """
    with open(os.path.join(model_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
        tfidf = pickle.load(f)

    with open(os.path.join(model_dir, 'product_vectors.pkl'), 'rb') as f:
        product_vectors = pickle.load(f)

    products = pd.read_csv(os.path.join(model_dir, 'products.csv'))

    return tfidf, product_vectors, products

def recommend_products(interaction_product_id, model_dir, top_n=5):
    """
    Générer des recommandations pour un produit donné.

    Args:
        interaction_product_id (int): ID du produit à partir duquel générer des recommandations.
        model_dir (str): Chemin du dossier contenant les fichiers du modèle.
        top_n (int): Nombre de recommandations à retourner.

    Returns:
        DataFrame: Les produits recommandés.
    """
    tfidf, product_vectors, products = load_model(model_dir)

    # Rechercher le produit par ID
    if interaction_product_id not in products['id_produit'].values:
        raise ValueError(f"Produit avec ID {interaction_product_id} non trouvé.")

    product_index = products[products['id_produit'] == interaction_product_id].index[0]

    # Calculer la similarité cosinus
    similarities = cosine_similarity(product_vectors[product_index], product_vectors).flatten()
    similar_indices = similarities.argsort()[::-1][1:top_n + 1]  # Exclure le produit lui-même

    recommended_products = products.iloc[similar_indices]
    recommended_products['similarity_score'] = similarities[similar_indices]

    return recommended_products

if __name__ == "__main__":
    model_dir = "ubuntu/desktop/mlops/model"
    interaction_product_id = 1  # Exemple : produit avec ID 1
    recommendations = recommend_products(interaction_product_id, model_dir, top_n=5)

    print("Recommandations :")
    print(recommendations)
