
import pandas as pd

from .config import MODEL_FEATURES, TEST_SIZE


def chronological_split(df, test_size=TEST_SIZE):

    # Trier les trajets par date
    ordered = df.sort_values(
        "tpep_pickup_datetime"
    ).reset_index(drop=True)

    if len(ordered) < 2:
        raise ValueError(
            "Il faut au moins deux trajets"
        )

    # Position de la séparation 80/20
    cutoff_idx = int(
        len(ordered) * (1 - test_size)
    )

    cutoff_idx = max(
        1,
        min(cutoff_idx, len(ordered) - 1)
    )

    cutoff_time = ordered.iloc[
        cutoff_idx
    ]["tpep_pickup_datetime"]

    # Trajets anciens : entraînement
    train_df = ordered.loc[
        ordered["tpep_pickup_datetime"] < cutoff_time
    ].copy()

    # Trajets récents : test
    test_df = ordered.loc[
        ordered["tpep_pickup_datetime"] >= cutoff_time
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Séparation chronologique impossible"
        )

    return train_df, test_df


def encode_features(train_df, test_df):

    # Features d'entraînement
    X_train = pd.get_dummies(
        train_df[MODEL_FEATURES],
        columns=["payment_type"],
        dtype=int
    )

    # Features de test
    X_test = pd.get_dummies(
        test_df[MODEL_FEATURES],
        columns=["payment_type"],
        dtype=int
    )

    # Garantir les mêmes colonnes
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    return X_train, X_test
