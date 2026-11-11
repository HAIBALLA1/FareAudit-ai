
from sklearn.ensemble import IsolationForest

from .config import RANDOM_STATE


def train_model(X_train):

    model = IsolationForest(
        n_estimators=100,
        contamination=0.02,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Entraînement uniquement sur le train
    model.fit(X_train)

    return model


def predict_test(model, X_test):

    # Prédiction sur les données non vues
    predictions = model.predict(X_test)

    # Convertir en booléens
    is_anomaly = predictions == -1

    # Score d'anomalie
    scores = model.decision_function(X_test)

    return is_anomaly, scores
