import json

import pandas as pd
from pydantic import BaseModel
from loguru import logger


class ListingPayload(BaseModel):
    neighborhood_group: str
    neighborhood: str
    room_type: str
    minimum_nights: int
    availability_365: int
    distance: float


def payload_to_df(payload):

    logger.info(payload)
    data = {
        "neighborhood_group": payload.neighborhood_group,
        "neighborhood": payload.neighborhood,
        "room_type": payload.room_type,
        "minimum_nights": payload.minimum_nights,
        "availability_365": payload.availability_365,
        "distance": payload.distance,
    }
    data = pd.DataFrame.from_dict([data])
    return data
