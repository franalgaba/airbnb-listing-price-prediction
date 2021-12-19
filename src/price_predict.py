import requests
import os
import json
from requests.models import Response

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
    """
    Builds Inference backend payload
    """

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

    st.title("Airbnbinator! El optimizador de precios en Airbnb :rocket:")
    st.markdown(
        "¡Llegó la hora de utilizar el producto final! :partying_face: Airbnbator te permitirá obtener el precio de listado óptimo para tu vivienda. Para empezar, introduce la dirección donde está tu vivienda:"
    )

    address = st.text_input("Dirección de Madrid", value="Plaza Mayor, Madrid")

    if address != "":
        geolocator = Nominatim(user_agent="custom")
        location = geolocator.geocode(address)
        df = pd.DataFrame(
            [(location.latitude, location.longitude)],
            columns=["lat", "lon"],
        )
        st.map(df)

    st.markdown(
        "Perfecto! :star_struck: Ahora rellena los últimos datos relativos a la vivienda:"
    )

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
            with st.spinner(":thinking: Consiguiendo el mejor precio de listado..."):
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
                    response = requests.post(
                        os.environ["PREDICTION_BACKEND_URL"], json=payload
                    )
                    prediction = json.loads(response.text)
                    st.success(
                        f'El precio por noche recomendado sería: __{prediction["price"]}__'
                    )
                else:
                    st.error("Missing PREDICTION_BACKEND_URL environment variable")

    st.header("Arquitectura")

    st.markdown(
        """
        Todo el desarrollo de este apartado no procesa las inferencias en el contenedor local. 
        
        Por el contrario, se ha productivizado el modelo de Keras entrenado previamente junto al One Hot Encoder del apartado anterior. Se ha desarrollado un microservicio productivo de inferencia en Python usando FastAPI para la capa de API y se ha construido un contenedor desplegado en el servicio serverless Cloud Run en GCP. 
        
        Los modelos entrenados están almacenados en GCS y este frontal lo que hace es realizar peticiones sobre este servicio para obtener las inferencias.
        """
    )


if __name__ == "__main__":
    write()
