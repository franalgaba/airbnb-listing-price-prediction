import pathlib

import pandas as pd
from geopy.distance import geodesic


def calculate_distance(longitude, latitude):

    """
    Calculates the distance in km to a reference point
    givem a longitude and latitude.
    """

    madrid = "40.416729, -3.703339"
    return geodesic(madrid, f"{longitude} {latitude}").km


def get_data():
    """
    Return lisiting data from Airbnb
    """
    return pd.read_csv(pathlib.Path(__file__).parent / "data/listings.csv")
