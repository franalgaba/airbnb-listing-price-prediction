import streamlit as st


def write():

    """
    Main entrypoint for the frontpage
    """

    st.title("Airbnb and chill... your house!")
    st.markdown(
        """
    Gracias a esta aplicación podrás explorar los datos de Airbnb en Madrid, sacar conclusiones de los datos entre más funcionalidades.

    Si estás aquí te estarás preguntando... ¿a qué precio pongo en alquiler mi vivienda? Utilizando Machine Learning podrás obtener el mejor precio al que listar tu vivienda al precio más competitivo.

    Mi nombre es Fran Algaba y estás en la aplicación interactiva que he diseñado para la prueba técnica de Keepler. La aplicación está dividida en tres secciones con las que podrás navegar desde el panel izquierdo:

    - EDA: en este apartado podrás explorar los datos de forma visual e intuitiva mientras te voy guiando con mis conclusiones de los datos.
    - Model Training: desde aquí podrás ver el procedimiento de entrenamiento de un modelo de Machine Learning para la predicción del mejor precio de listado para viviendas en Airbnb.
    - Model Deployment: en este apartado podrás probar el producto final desarrollado. Dada una dirección cualquiera de Madrid podrás obtner predicciones de precio. Todo esto montando en una arquitectura serverless sobre GCP donde se ha productivizado el modelo para que este frontal le haga consultas.

    Todo el código de la aplicación está disponible en: [airbnb-listing-price-prediction](https://github.com/franalgaba/airbnb-listing-price-prediction)
    También puedes ver mis perfiles de redes sociales:
    - Linkedin: xxx
    - Github: xxx
    - Twitter donde sobretodo hablo de mi nuevo interés, la web3: xxx
    """
    )


if __name__ == "__main__":
    write()
