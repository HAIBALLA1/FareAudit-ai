import pytest

from src.fareaudit.schemas import TripInput
from src.fareaudit.inference import predict_trip


def test_valid_trip():
    trip = TripInput(
        trip_distance=12.5,
        fare_amount=45,
        tip_amount=5,
        total_amount=55,
        pickup_hour=14,
        payment_type=1,
    )

    assert trip.pickup_hour == 14

