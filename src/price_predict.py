import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim


def write():

    st.title("Airbnb listing price prediction! :rocket:")
    st.markdown("Por favor, inserta la dirección que quieras predecir:")

    address = st.text_input("Dirección de Madrid")

    if address != "":
        geolocator = Nominatim(user_agent="custom")
        location = geolocator.geocode(address)
        st.text(location)
        df = pd.DataFrame(
            [(location.latitude, location.longitude)],
            columns=["lat", "lon"],
        )
        st.map(df)


if __name__ == "__main__":
    write()
