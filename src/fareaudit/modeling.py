from pandas import DataFrame
from sklearn.ensemble import IsolationForest

from .config import RANDOM_STATE


N_ESTIMATORS = 100
CONTAMINATION = 0.02


def train_model(X_train: DataFrame) -> IsolationForest:
    """Entraîne un modèle Isolation Forest."""

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train)

    return model


def predict_test(model: IsolationForest, X_test: DataFrame):
    """Retourne les indicateurs d'anomalie et les scores."""

    predictions = model.predict(X_test)
    is_anomaly = predictions == -1
    scores = model.decision_function(X_test)

    return is_anomaly, scores