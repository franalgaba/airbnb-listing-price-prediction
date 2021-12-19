import pathlib

import pandas as pd


def get_data():
    return pd.read_csv(pathlib.Path(__file__).parent / "data/listings.csv")
