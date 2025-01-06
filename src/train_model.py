import pandas as pd
import mlflow
import mlflow.sklearn
from custom_model import CustomModel

# Charger les fichiers CSV
products = pd.read_csv('../../web/data/products.csv')
products['text_features'] = products['main_category'] + " " + products['sub_category'] + " " + products['name']

# Créer et entraîner CustomModel
model = CustomModel()
model.fit(products['text_features'])

# Exemple d'entrée pour l'enregistrement
input_example = products['text_features'].iloc[0:1].tolist()

# Enregistrer le modèle avec MLflow
def train_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("my_experiment")

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "custom_model", input_example={"text_features": input_example})
        mlflow.log_param("num_products", len(products))
        mlflow.log_metric("tfidf_vocab_size", len(model.vectorizer.vocabulary_))
        print(f"Run ID: {run.info.run_id}")

    print("Modèle personnalisé entraîné et enregistré avec MLflow.")

if __name__ == '__main__':
    train_model()

