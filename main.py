from src.data_preprocessing import load_all_datasets, prepare_data

# Chemin vers le dossier contenant les fichiers CSV
data_dir = "data/"

# Chargement et combinaison des datasets
data = load_all_datasets(data_dir)

if data is not None:
    # Prétraitement des données
    data = prepare_data(data)
    print(data.head())
else:
    print("Aucun dataset à traiter.")
