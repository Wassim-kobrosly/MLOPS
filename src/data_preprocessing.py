import os
import pandas as pd

def load_data(data_dir):
    """
    Charger tous les datasets depuis le dossier donné.

    Args:
        data_dir (str): Chemin du dossier contenant les datasets.

    Returns:
        dict: Un dictionnaire contenant les DataFrames des datasets chargés.
    """
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    datasets = {}

    for file in data_files:
        file_path = os.path.join(data_dir, file)
        dataset_name = os.path.splitext(file)[0]
        datasets[dataset_name] = pd.read_csv(file_path)

    return datasets

def preprocess_data(datasets):
    """
    Prétraiter les données pour les rendre prêtes pour le modèle de recommandation.

    Args:
        datasets (dict): Dictionnaire contenant les DataFrames des datasets chargés.

    Returns:
        dict: Dictionnaire contenant les datasets prétraités.
    """
    users = datasets.get('users', None)
    products = datasets.get('products', None)
    interactions = datasets.get('interactions', None)

    if users is None or products is None or interactions is None:
        raise ValueError("Les fichiers 'users.csv', 'products.csv' et 'interactions.csv' sont requis.")

    # Ajouter un ID d'utilisateur si nécessaire
    if 'user_id' not in users.columns:
        users['user_id'] = range(1, len(users) + 1)

    # Ajouter un ID de produit si nécessaire
    if 'product_id' not in products.columns:
        products['product_id'] = range(1, len(products) + 1)

    # Filtrer les interactions pour inclure uniquement les utilisateurs et les produits existants
    interactions = interactions[interactions['user_id'].isin(users['user_id'])]
    interactions = interactions[interactions['product_id'].isin(products['product_id'])]

    # Ajouter un ID d'interaction si nécessaire
    if 'interaction_id' not in interactions.columns:
        interactions['interaction_id'] = range(1, len(interactions) + 1)

    datasets['users'] = users
    datasets['products'] = products
    datasets['interactions'] = interactions

    return datasets

if __name__ == "__main__":
    data_dir = "../../test/data"  # Chemin du répertoire contenant les fichiers CSV
    data_save = "../data"
    datasets = load_data(data_dir)
    processed_datasets = preprocess_data(datasets)

    # Sauvegarder les fichiers prétraités
    processed_datasets['users'].to_csv(os.path.join(data_save, 'processed_users.csv'), index=False)
    processed_datasets['products'].to_csv(os.path.join(data_save, 'processed_products.csv'), index=False)
    processed_datasets['interactions'].to_csv(os.path.join(data_save, 'processed_interactions.csv'), index=False)

    print("Prétraitement terminé et fichiers sauvegardés.")

