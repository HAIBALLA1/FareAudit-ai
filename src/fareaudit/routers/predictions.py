from fastapi import APIRouter
from ..schemas import TripInput , PredictionOutput

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
)

@router.get("/predict" , response_model=PredictionOutput)
def predict(data: TripInput):
    pass