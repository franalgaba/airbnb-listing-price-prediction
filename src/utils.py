import pathlib

import pandas as pd
from geopy.distance import geodesic


def calculate_distance(self, longitude, latitude):
    madrid = "40.416729, -3.703339"
    return geodesic(madrid, f"{longitude} {latitude}").km


def get_data():
    return pd.read_csv(pathlib.Path(__file__).parent / "data/listings.csv")
