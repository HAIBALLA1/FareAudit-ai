import pytest
from pydantic import ValidationError

from src.fareaudit.schemas import TripInput , PredictionOutput

VALID_TRIP ={
    "trip_distance" :12.5,
    "fare_amount" : 45,
    "tip_amount" : 5,
    "total_amount" : 55,
    "pickup_hour" : 14,
    "payment_type" : 1
}

INFR_RESULT ={
             "prediction": -1,
             "is_anomaly": True,
             "anomaly_score": -0.08,
            }



def test_valid_trip():
    trip = TripInput(**VALID_TRIP)

    assert trip.model_dump() == VALID_TRIP


def test_extra_field_forbidden():
    data={
        **VALID_TRIP,
        "trip_tem" : 12.5,
    }

    with pytest.raises(ValidationError):
        TripInput(**data)


@pytest.mark.parametrize(
    "field",
    [
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "total_amount",
        "pickup_hour",
        "payment_type",
    ]
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        float("inf"),
        float("-inf"),
        float("nan"),
    ]
)
def test_infinit_field_forbidden(invalid_value ,field ):
    data={
        **VALID_TRIP,
        field : invalid_value,
    }

    with pytest.raises(ValidationError):
        TripInput(**data)


def test_PredictionOutput():
    model_output = PredictionOutput(**INFR_RESULT)

    assert model_output.model_dump() == INFR_RESULT

def test_TripOutput():
    data={
        **INFR_RESULT,
        "prediction": 5,
    }

    with pytest.raises(ValueError):
        PredictionOutput(**data)