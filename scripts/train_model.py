import joblib

from src.fareaudit.config import DATA_PATH, MODEL_PATH, SAMPLE_SIZE
from src.fareaudit.data_loader import load_sample
from src.fareaudit.data_preprocessing import prepare_data
from src.fareaudit.features import chronological_split, encode_features
from src.fareaudit.modeling import train_model, predict_test



def save_model(model, X_train, train_df):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": X_train.columns.tolist(),
            "train_end_date": str(train_df["tpep_pickup_datetime"].max()),
        },
        MODEL_PATH,
    )


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

    df = load_sample(DATA_PATH, sample_size=SAMPLE_SIZE)
    df = prepare_data(df)

    train_df, test_df = chronological_split(df)

    if train_df["tpep_pickup_datetime"].max() >= test_df["tpep_pickup_datetime"].min():
        raise ValueError("Chevauchement temporel entre train et test")

    X_train, X_test = encode_features(train_df, test_df)

    model = train_model(X_train)
    anomalies, scores = predict_test(model, X_test)

    results_df = test_df.assign(
        anomaly_ml=anomalies,
        anomaly_score=scores,
    )

    save_model(model, X_train, train_df)


if __name__ == "__main__":
    main()
