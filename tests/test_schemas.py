import pytest
from pydantic import ValidationError

from src.fareaudit.schemas import TripInput

VALID_TRIP ={
    "trip_distance" :12.5,
    "fare_amount" : 45,
    "tip_amount" : 5,
    "total_amount" : 55,
    "pickup_hour" : 14,
    "payment_type" : 1
}


def test_valid_trip():
    trip = TripInput(**VALID_TRIP)

    assert trip.pickup_hour == 14
    assert trip.payment_type == 1
    assert trip.trip_distance == 12.5
    assert trip.fare_amount == 45
    assert trip.total_amount == 55
    assert trip.tip_amount == 5


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