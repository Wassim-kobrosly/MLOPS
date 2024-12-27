# data_preprocessing.py

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
    products = datasets.get('merged_dataset_cleaned', None)
    interactions = datasets.get('interactions', None)

    if products is None or interactions is None:
        raise ValueError("Les fichiers 'merged_dataset_cleaned.csv' et 'interactions.csv' sont requis.")

    # Ajouter un ID de produit si nécessaire
    if 'id_produit' not in products.columns:
        products['id_produit'] = range(1, len(products) + 1)

    # Filtrer les interactions pour inclure uniquement les produits existants
    interactions = interactions[interactions['product_id'].isin(products['id_produit'])]

    # Ajouter un ID d'interaction si nécessaire
    if 'interaction_id' not in interactions.columns:
        interactions['interaction_id'] = range(1, len(interactions) + 1)

    datasets['products'] = products
    datasets['interactions'] = interactions

    return datasets

if __name__ == "__main__":
    data_dir = "~/desktop/mlops/data"
    datasets = load_data(data_dir)
    processed_datasets = preprocess_data(datasets)

    # Sauvegarder les fichiers prétraités
    processed_datasets['products'].to_csv(os.path.join(data_dir, 'processed_products.csv'), index=False)
    processed_datasets['interactions'].to_csv(os.path.join(data_dir, 'processed_interactions.csv'), index=False)

    print("Prétraitement terminé et fichiers sauvegardés.")
