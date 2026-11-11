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

from src.fareaudit.reporting import print_report


def main():

    # ETAPE 1 : Chargement des données
    print("\n1. Chargement des données...")

    print("Chemin du dataset :", DATA_PATH)

    # Vérifier que le fichier existe
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

    print("X_train :", X_train.shape)
    print("X_test :", X_test.shape)

    print(
        "Dernière date du train :",
        train_df["tpep_pickup_datetime"].max(),
    )

    print(
        "Première date du test :",
        test_df["tpep_pickup_datetime"].min(),
    )

    # Vérifier la séparation temporelle
    assert (
        train_df["tpep_pickup_datetime"].max()
        < test_df["tpep_pickup_datetime"].min()
    ), "Erreur : chevauchement temporel train/test"

    # ETAPE 4 : Entraînement
    print("\n4. Entraînement Isolation Forest...")

    model = train_model(X_train)

    print("Entraînement terminé")

    print("Seuil du modèle :", model.offset_)

    # ETAPE 5 : Prédictions sur X_test
    print("\n5. Prédictions sur les données de test...")

    anomalies, scores = predict_test(
        model,
        X_test,
    )

    # Conserver les résultats dans le test
    test_df = test_df.copy()

    test_df["anomaly_ml"] = anomalies
    test_df["anomaly_score"] = scores

    print(
        "Nombre d'anomalies ML :",
        test_df["anomaly_ml"].sum(),
    )

    # ETAPE 6 : Analyse des résultats
    print("\n6. Rapport des anomalies...")

    print_report(test_df)

    # ETAPE 7 : Sauvegarde du modèle
    print("\n7. Sauvegarde du modèle...")

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

    print("\nPipeline ML terminé avec succès !")


if __name__ == "__main__":
    main()
