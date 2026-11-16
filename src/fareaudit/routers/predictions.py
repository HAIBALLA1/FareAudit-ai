from fastapi import APIRouter
from ..schemas import TripInput , PredictionOutput
from ..services.prediction_service import prediction

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
)

@router.post("/" , response_model=PredictionOutput)
def predict(data: TripInput):
    return prediction(data)