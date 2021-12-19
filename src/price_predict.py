import requests
import os

import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim

from src.utils import get_data, calculate_distance


def _build_predict_request(
    neighborhood_group,
    neighborhood,
    room_type,
    minimum_nights,
    availability_365,
    latitude,
    longitude,
):
    st.text(
        f"{neighborhood_group}, {neighborhood}, {room_type}, {minimum_nights}, {availability_365}, {latitude}, {longitude}"
    )

    distance = calculate_distance(longitude, latitude)

    return {
        "neighborhood_group": neighborhood_group,
        "neighborhood": neighborhood,
        "room_type": room_type,
        "minimum_nights": minimum_nights,
        "availability_365": availability_365,
        "distance": distance,
    }


def write():

    st.title("Airbnb listing price prediction! :rocket:")
    st.markdown("Por favor, inserta la dirección que quieras predecir:")

    address = st.text_input("Dirección de Madrid", value="Plaza Mayor, Madrid")

    if address != "":
        geolocator = Nominatim(user_agent="custom")
        location = geolocator.geocode(address)
        df = pd.DataFrame(
            [(location.latitude, location.longitude)],
            columns=["lat", "lon"],
        )
        st.map(df)

    st.markdown("Ahora introduce la información relativa a la propiedad:")

    df = get_data()

    with st.form("Prediction Data"):
        neighborhood_group = st.selectbox("Zona", df.neighbourhood_group.unique())
        neighborhood = st.selectbox("Barrio", df.neighbourhood.unique())
        room_type = st.selectbox("Tipo de alojamiento", df.room_type.unique())
        minimum_nights = st.number_input("Número mínimo de noches", min_value=1)
        availability_365 = st.number_input(
            "Días disponibles al año", min_value=minimum_nights, max_value=365
        )

        submitted = st.form_submit_button("Get listing price")
        if submitted:
            payload = _build_predict_request(
                neighborhood_group,
                neighborhood,
                room_type,
                minimum_nights,
                availability_365,
                location.latitude,
                location.longitude,
            )
            if "PREDICTION_BACKEND_URL" in os.environ:
                prediction = requests.post("PREDICTION_BACKEND_URL", json=payload)
                st.success(
                    f'El precio por noche recomendado sería: __{prediction["price"]}__'
                )
            else:
                st.error("Missing PREDICTION_BACKEND_URL environment variable")


if __name__ == "__main__":
    write()
