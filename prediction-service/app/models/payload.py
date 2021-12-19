import json

import pandas as pd
from pydantic import BaseModel


class ListingPayload(BaseModel):
    neighborhood_group: str
    neighborhood: str
    room_type: str
    minimum_nights: int
    availability_365: int
    distance: float


def payload_to_df(payload):

    data = json.loads(payload.text)
    return pd.DataFrame.from_dict([data])
