from ..inference import predict_trip


def prediction(trip) -> dict:

    result=predict_trip(trip)

    return result