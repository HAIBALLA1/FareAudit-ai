
import numpy as np
import pandas as pd

from .config import PAYMENT_LABELS


def prepare_data(df):

    # Ne pas modifier le DataFrame original
    data = df.copy()

    # Convertir la date en datetime
    data["tpep_pickup_datetime"] = pd.to_datetime(
        data["tpep_pickup_datetime"],
        errors="coerce"
    )

    # Colonnes numériques
    numeric_columns = [
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "total_amount",
        "payment_type"
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Remplacer les valeurs infinies
    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Supprimer les lignes invalides
    data = data.dropna(
        subset=[
            "tpep_pickup_datetime",
            *numeric_columns
        ]
    ).copy()

    # Conserver janvier 2025 uniquement
    pickup = data["tpep_pickup_datetime"]

    data = data.loc[
        (pickup.dt.year == 2025)
        & (pickup.dt.month == 1)
    ].copy()

    if data.empty:
        raise ValueError(
            "Aucun trajet valide pour janvier 2025"
        )

    # Type de paiement
    data["payment_type"] = (
        data["payment_type"].astype(int)
    )

    data["payment_label"] = (
        data["payment_type"]
        .map(PAYMENT_LABELS)
        .fillna("Autre")
    )

    # Extraire l'heure
    data["pickup_hour"] = (
        data["tpep_pickup_datetime"].dt.hour
    )

    # Règles exploratoires
    data["negative_total"] = (
        data["total_amount"] < 0
    )

    data["zero_distance"] = (
        data["trip_distance"] == 0
    )

    data["high_amount"] = (
        data["total_amount"] > 200
    )

    # Au moins une règle déclenchée
    flags = [
        "negative_total",
        "zero_distance",
        "high_amount"
    ]

    data["anomaly_rules"] = (
        data[flags].any(axis=1)
    )

    return data
