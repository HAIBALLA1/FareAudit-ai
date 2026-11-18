from unittest.mock import Mock
from sklearn.ensemble import IsolationForest
import pandas as pd
from src.fareaudit import inference
from src.fareaudit.schemas import TripInput

VALID_TRIP ={
    "trip_distance" :12.5,
    "fare_amount" : 45,
    "tip_amount" : 5,
    "total_amount" : 55,
    "pickup_hour" : 14,
    "payment_type" : 1
}

def test_predict_trip_anomaly(monkeypatch):
    trip = TripInput(**VALID_TRIP)

    #un faux objet qui va imiter mon modèle
    fake_model=Mock(spec=IsolationForest)

    #definir le comportement de mon fake model
    fake_model.predict.return_value = [-1]
    fake_model.decision_function.return_value = [-0.08]

    monkeypatch.setattr(inference,
                        "load_model" ,
                        lambda: {"model" :fake_model,
                                 "feature_columns": []}
    )
    fake_X = pd.DataFrame([])
    monkeypatch.setattr(inference,"prepare_trip_features",
                        lambda trip, feature_columns: fake_X)

    result = inference.predict_trip(trip)
    assert result ==  {
                            "prediction": -1,
                            "is_anomaly": True,
                            "anomaly_score": -0.08,
                        }

