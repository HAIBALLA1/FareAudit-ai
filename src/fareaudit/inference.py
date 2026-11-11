
from functools import lru_cache

import joblib
import pandas as pd

from .config import MODEL_PATH
from .schemas import TripInput


# Charger le modèle une seule fois par processus
@lru_cache(maxsize=1)
def load_model():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    return bundle


def predict_trip(trip: TripInput):

    # 1. Charger le modèle entraîné
    bundle = load_model()

    model = bundle["model"]

    feature_columns = bundle["feature_columns"]

    # 2. Convertir les données en dictionnaire
    trip_data = trip.model_dump()

    # 3. Récupérer le type de paiement
    payment_type = trip_data.pop("payment_type")

    payment_column = f"payment_type_{payment_type}"

    # Vérifier que cette catégorie était présente
    # pendant l'entraînement
    if payment_column not in feature_columns:
        raise ValueError(
            f"Type de paiement {payment_type} "
            "non présent dans les données "
            "d'entraînement du modèle"
        )

    # 4. Créer un DataFrame d'une ligne
    X = pd.DataFrame([trip_data])

    # 5. One-Hot Encoding
    X[payment_column] = 1

    # 6. Reproduire exactement les colonnes
    # utilisées pendant l'entraînement
    X = X.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # 7. Prédiction
    prediction = int(model.predict(X)[0])

    # 8. Score d'anomalie
    score = float(
        model.decision_function(X)[0]
    )

    # 9. Construire la réponse
    return {
        "prediction": prediction,
        "is_anomaly": prediction == -1,
        "anomaly_score": score
    }
