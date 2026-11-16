from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from .config import MODEL_PATH

@lru_cache(maxsize=1)
def load_model() -> dict[str, Any]:
    """Charge le modèle sauvegardé une fois par processus."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def prepare_trip_features(
    trip: TripInput,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Transforme une course dans le format attendu par le modèle."""

    trip_data = trip.model_dump()

    payment_type = trip_data.pop("payment_type")
    payment_column = f"payment_type_{payment_type}"

    if payment_column not in feature_columns:
        raise ValueError(
            f"Type de paiement non reconnu par le modèle : "
            f"{payment_type}"
        )

    X = pd.DataFrame([trip_data])

    X[payment_column] = 1

    return X.reindex(
        columns=feature_columns,
        fill_value=0,
    )


def predict_trip(trip) -> dict:
    """Prédit si une course est anormale."""

    donnees_modele = load_model()

    model = donnees_modele["model"]
    feature_columns = donnees_modele["feature_columns"]

    X = prepare_trip_features(trip, feature_columns)

    prediction = int(model.predict(X)[0])
    score = float(model.decision_function(X)[0])

    return {
        "prediction": prediction,
        "is_anomaly": prediction == -1,
        "anomaly_score": score,
    }