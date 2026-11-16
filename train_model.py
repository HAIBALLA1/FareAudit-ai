import joblib

from src.fareaudit.config import (
    DATA_PATH,
    MODEL_PATH,
    SAMPLE_SIZE,
)

from src.fareaudit.data_loader import load_sample
from src.fareaudit.data_preprocessing import prepare_data

from src.fareaudit.features import (
    chronological_split,
    encode_features,
)

from src.fareaudit.modeling import (
    train_model,
    predict_test,
)


def main():

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}"
        )

    df = load_sample(
        DATA_PATH,
        sample_size=SAMPLE_SIZE,
    )

    print("Trajets chargés :", len(df))

    # ETAPE 2 : Data Preprocessing
    print("\n2. Préparation des données...")

    df = prepare_data(df)

    print("Trajets conservés :", len(df))

    # ETAPE 3 : Séparation Train / Test
    print("\n3. Séparation chronologique...")

    train_df, test_df = chronological_split(df)

    X_train, X_test = encode_features(
        train_df,
        test_df,
    )

    # Vérifier la séparation temporelle
    assert (
        train_df["tpep_pickup_datetime"].max()
        < test_df["tpep_pickup_datetime"].min()
    ), "Erreur : chevauchement temporel train/test"

    model = train_model(X_train)

    anomalies, scores = predict_test(
        model,
        X_test,
    )

    test_df = test_df.copy()

    test_df["anomaly_ml"] = anomalies
    test_df["anomaly_score"] = scores

    print(
        "Nombre d'anomalies ML :",
        test_df["anomaly_ml"].sum(),
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": X_train.columns.tolist(),
            "train_end_date": str(
                train_df["tpep_pickup_datetime"].max()
            ),
        },
        MODEL_PATH,
    )

    print("Modèle sauvegardé :", MODEL_PATH)


if __name__ == "__main__":
    main()
