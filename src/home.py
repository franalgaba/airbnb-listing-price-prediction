import streamlit as st


def write():
    st.title("Airbnb and chill... your house!")
    st.markdown(
        """
    Gracias a esta aplicación podrás explorar los datos de Airbnb en Madrid así como sacar conclusiones de los datos de forma intuitiva.

    Si estás aquí te estarás preguntando... ¿a qué precio pongo en alquiler mi vivienda? Utilizando Machine Learning podrás obtener el mejor precio al que listar tu vivienda con el precio más competitivo.

    Mi nombre es Fran Algaba y estás en la aplicación interactiva que he diseñado para la prueba técnica de Keepler (¡hola Ramiro!). La aplicación está dividida en dos secciones con las que podrás navegar desde el panel izquierdo:

    - EDA: en este apartado podrás explorar los datos de forma visual e intuitiva mientras te voy guiando con mis conclusiones de los datos.
    - Review Analysis: aquí podrás ver un análisis de sentimiento usando técnicas de NLP sobre las reviews para ofrecer un análisis de las mejores zonas en Madrid para alojarse.
    - Listing Price Training: desde aquí podrás ver el procedimiento de entrenamiento de un modelo de Machine Learning para la predicción del mejor precio de listado para viviendas en Airbnb.
    - Listing Price Prediction: en este apartado dado una vivienda que se quiera poner en venta, se ofrecerá el mejor precio de listado en base a la competencia de los alrededores.

    Todo el código de la aplicación está disponible en: xxx
    También puedes ver mis perfiles de redes sociales:
    - Linkedin: xxx
    - Github: xxx
    - Twitter donde sobretodo hablo de mi nuevo interés, la web3: xxx
    """
    )


if __name__ == "__main__":
    write()
