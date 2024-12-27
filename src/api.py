from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialisation de l'application Flask
app = Flask(__name__)

# Charger les données et le modèle
DATA_PATH = 'ubuntu/desktop/mlops/data/'
products = pd.read_csv(f'{DATA_PATH}merged_dataset_cleaned.csv')

# Prétraitement des données produits
products['text_features'] = products['main_category'] + " " + products['sub_category'] + " " + products['name']
tfidf = TfidfVectorizer()
product_vectors = tfidf.fit_transform(products['text_features'])

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Point de terminaison pour obtenir des recommandations.
    """
    data = request.json
    product_id = data.get('product_id')
    action_type = data.get('action', 'viewed')  # Action par défaut : 'viewed'

    # Vérifier si le produit existe
    if product_id not in products['id_produit'].values:
        return jsonify({"error": "Produit non trouvé"}), 404

    last_product = products[products['id_produit'] == product_id].iloc[0]

    if action_type == "purchased":
        # Recommander un produit d'une autre sous-catégorie mais de la même catégorie
        recommendations = products[
            (products['main_category'] == last_product['main_category']) &
            (products['sub_category'] != last_product['sub_category'])
        ].head(5)
    else:
        # Recommander des produits similaires basés sur la similarité cosinus
        product_index = products[products['id_produit'] == product_id].index[0]
        similarities = cosine_similarity(product_vectors[product_index], product_vectors).flatten()
        similar_indices = similarities.argsort()[::-1][1:6]  # Obtenir les 5 produits les plus similaires
        recommendations = products.iloc[similar_indices]

    # Construire la réponse
    response = recommendations[['id_produit', 'name', 'main_category', 'sub_category']].to_dict(orient='records')
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
