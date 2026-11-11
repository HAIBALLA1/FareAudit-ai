from fastapi import APIRouter, HTTPException

from .schemas import (
    TripInput,
    PredictionOutput
)

from .inference import predict_trip


router = APIRouter(
    tags=["Machine Learning"]
)


@router.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(trip: TripInput):

    try:
        result = predict_trip(trip)

        return result

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        ) from error
