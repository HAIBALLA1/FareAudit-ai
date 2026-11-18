from typing import Literal
from pydantic import BaseModel, Field, ConfigDict , model_validator


class TripInput(BaseModel):

    #on veut interdire les champs extra et les valeurs infini ou nane
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
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False
    )

    prediction: Literal[-1, 1]
    is_anomaly: bool
    anomaly_score: float

    @model_validator(mode="after")
    def check_prediction_consistency(self):
        if self.is_anomaly != (self.prediction == -1):
            raise ValueError(
                "is_anomaly is inconsistent with prediction"
            )

        return self