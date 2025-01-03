import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Charger les fichiers CSV
products = pd.read_csv('../../test/data/products.csv')
users = pd.read_csv('../../test/data/users.csv')
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

# Fonction de recommandation
def recommend_and_store_prediction(interactions, products, product_vectors):
    recommendations = []

    for _, interaction in interactions.iterrows():
        # Récupérer les informations de l'interaction
        last_product_id = interaction['product_id']
        action_type = interaction['action']

        # Vérifier si le produit de l'interaction existe
        if products[products['id_produit'] == last_product_id].empty:
            continue
        last_product = products[products['id_produit'] == last_product_id].iloc[0]

        # Si l'action est un achat, recommander un produit d'une autre sous-catégorie mais de la même catégorie
        if action_type == "purchased":
            recommendations_filtered = products[
                (products['main_category'] == last_product['main_category']) &
                (products['sub_category'] != last_product['sub_category'])
            ]
        else:
            # Sinon, recommander des produits similaires basés sur la similarité cosinus
            product_index = products[products['id_produit'] == last_product_id].index[0]
            similarities = cosine_similarity(product_vectors[product_index], product_vectors).flatten()
            similar_indices = similarities.argsort()[::-1][1:6]  # Obtenir les 5 produits les plus similaires
            recommendations_filtered = products.iloc[similar_indices]

        # Limiter à 5 recommandations
        recommendations_filtered = recommendations_filtered.head(5)

        # Générer des recommandations
        for _, recommended_product in recommendations_filtered.iterrows():
            id_prediction = len(recommendations) + 1  # Générer un ID simple incrémental
            id_interaction = interaction['interaction_id']
            name_product_interaction = last_product['name']  # Nom du produit de l'interaction
            name_product_recommendation = recommended_product['name']  # Nom du produit recommandé
            taux_prediction = 1.0 if action_type == "purchased" else cosine_similarity(
                product_vectors[products[products['id_produit'] == recommended_product['id_produit']].index[0]],
                product_vectors[product_index]
            ).flatten()[0]

            recommendations.append({
                'id_prediction': id_prediction,
                'id_interaction': id_interaction,
                'action': action_type,
                'name_product_interaction': name_product_interaction,
                'id_produit_recommande': recommended_product['id_produit'],
                'name_produit_recommande': name_product_recommendation,
                'main_category': recommended_product['main_category'],
                'sub_category': recommended_product['sub_category'],
                'taux_prediction': taux_prediction
            })

    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df.to_csv('../../test/data/predictions.csv', index=False, mode='a', header=not pd.io.common.file_exists('../../test/data/predictions.csv'))

    return recommendations_df

# Générer les recommandations pour toutes les interactions
recommendations = recommend_and_store_prediction(interactions, products, product_vectors)


