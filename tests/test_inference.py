from unittest.mock import Mock
from sklearn.ensemble import IsolationForest
import pandas as pd
from src.fareaudit import inference
from src.fareaudit.schemas import TripInput
import pytest


VALID_TRIP ={
    "trip_distance" :12.5,
    "fare_amount" : 45,
    "tip_amount" : 5,
    "total_amount" : 55,
    "pickup_hour" : 14,
    "payment_type" : 1
}

FEATURE_COLUMNS = [
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "pickup_hour",
    "payment_type_1",
    "payment_type_2",
    "payment_type_3",
    "payment_type_4",
]

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

    assert result ==  {     "prediction": -1,
                            "is_anomaly": True,
                            "anomaly_score": -0.08,
                        }

    fake_model.predict.assert_called_once_with(fake_X)
    fake_model.decision_function.assert_called_once_with(fake_X)

#elle sert plutot a rien vu que ca va etre decaler par pydantic avant que ca arrive a notre methode
def test_prepare_trip_features():
    data = {**VALID_TRIP , "payment_type" : 7}

    with pytest.raises(ValueError):
        inference.prepare_trip_features(TripInput(**data) , feature_columns)