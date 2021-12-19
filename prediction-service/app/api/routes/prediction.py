from fastapi import APIRouter
from starlette.requests import Request

from app.models.payload import ListingPayload
from app.models.prediction import ListingPredictionResult
from app.services.models import ListingPriceModel

router = APIRouter()


@router.post("/predict", response_model=ListingPredictionResult, name="predict")
def post_predict(
    request: Request,
    data: ListingPayload = None,
) -> ListingPredictionResult:

    model: ListingPriceModel = request.app.state.model
    prediction: ListingPredictionResult = model.predict(data)

    return prediction
