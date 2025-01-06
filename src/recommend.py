import pandas as pd
from custom_model import CustomModel
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.sklearn
import os

# Charger les fichiers CSV
products = pd.read_csv('../../web/data/products.csv')
interactions = pd.read_csv('../../web/data/interactions.csv')

# Ajouter une colonne 'id_produit' si elle n'existe pas
if 'id_produit' not in products.columns:
    products['id_produit'] = range(1, len(products) + 1)

# Prétraitement des données produits
products['text_features'] = products['main_category'] + " " + products['sub_category'] + " " + products['name']

# Utiliser CustomModel pour transformer les données
model = CustomModel()
product_vectors = model.fit_transform(products['text_features'])

# Traiter uniquement la dernière interaction ajoutée
last_interaction = interactions.iloc[[-1]]

def recommend_and_store_prediction(last_interaction, products, product_vectors):
    recommendations = []

    with mlflow.start_run():
        for _, interaction in last_interaction.iterrows():
            with mlflow.start_run(nested=True):
                last_product_id = interaction['product_id']
                action_type = interaction['action']

                if products[products['id_produit'] == last_product_id].empty:
                    continue
                last_product = products[products['id_produit'] == last_product_id].iloc[0]

                if action_type == "purchased":
                    recommendations_filtered = products[
                        (products['main_category'] == last_product['main_category']) &
                        (products['sub_category'] != last_product['sub_category'])
                    ]
                else:
                    product_index = products[products['id_produit'] == last_product_id].index[0]
                    similarities = cosine_similarity(product_vectors[product_index], product_vectors).flatten()
                    similar_indices = similarities.argsort()[::-1][1:6]
                    recommendations_filtered = products.iloc[similar_indices]

                recommendations_filtered = recommendations_filtered.head(5)

                for _, recommended_product in recommendations_filtered.iterrows():
                    id_prediction = len(recommendations) + 1
                    id_interaction = interaction['interaction_id']
                    name_product_interaction = last_product['name']
                    name_product_recommendation = recommended_product['name']
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

                    mlflow.log_param("user_id", interaction['user_id'])
                    mlflow.log_param("product_id", last_product_id)
                    mlflow.log_metric("taux_prediction", taux_prediction)

    recommendations_df = pd.DataFrame(recommendations)
    file_path = '../../web/data/predictions.csv'
    file_exists = os.path.isfile(file_path)
    
    recommendations_df.to_csv(file_path, index=False, mode='a', header=not file_exists)
    print("Les recommandations ont été ajoutées à predictions.csv")

    return recommendations_df

if __name__ == '__main__':
    recommendations = recommend_and_store_prediction(last_interaction, products, product_vectors)

