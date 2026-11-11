
from pydantic import BaseModel, Field, ConfigDict


class TripInput(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False
    )

    trip_distance: float
    fare_amount: float
    tip_amount: float
    total_amount: float

    pickup_hour: int = Field(
        ge=0,
        le=23
    )

    payment_type: int = Field(
        ge=0,
        le=6
    )


class PredictionOutput(BaseModel):

    prediction: int
    is_anomaly: bool
    anomaly_score: float
