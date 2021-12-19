from pydantic import BaseModel


class ListingPredictionResult(BaseModel):
    price: float
    elapsed_time: float
