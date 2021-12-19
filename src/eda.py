import streamlit as st
import numpy as np
from scipy import stats

from src.utils import get_data


def write():

    df = get_data()

    st.title("Airbnb Exploratory Data Analysis")

    st.markdown(
        "Lo primero, vamos a familiarizarnos con el dataset, sus columnas y estructura. A continuación se muestran las primeras 10 filas de los datos:"
    )

    st.dataframe(df.head(10))

    st.header("Let's map some data!")

    st.markdown(
        "A continuación, vamos a ver es una representación visual de los pisos listados en Madrid utilizando su información geográfica. De forma interactiva en el mapa podremos filtrar la información en base a cuatro parámetros: el rango de precio, el tiempo de alojamiento, el número de reviews y el tipo de alojamiento."
    )

    st.markdown(
        """
        Si establecemos el número de noches en `7` para estancia de una semana y vamos variando el rango de precios, vemos como la mayor oferta en esta categoría se situa en la zona de Madrid centro, en los barrios de Malasaña, Chueca y Lavapies, estancias muy céntricas donde los visitantes a la capital tendrán más posibilidades de elección. 
        
        Sin embargo, hay otra zona que destaca, la zona de Usera tiene una gran concentración de oferta para un rango de precios bajo, reflejando como puede ser la opción elegida para los bolsillos más contenidos. Si en el mismo período de estancia de una semana vamos incrementando el rango de precio vemos como hay un cambio de tendencia, la zona centro mencionada sigue teniendo mucha oferta pero disminuye considerablemente mientras otros barrios como Tetuán, Cuatro Caminos y San Blas recogen el testigo como zonas de alojamiento para bolsillos más holgados.
        """
    )

    price_range = st.slider(
        "Rango de precio ($)",
        float(df.price.min()),
        float(df.price.max()),
        (500.0, 1500.0),
    )
    num_nights = st.slider("Número de noches", 0, 30, (7))
    num_reviews = st.slider("Número de reseñas", 0, 700, (0))
    room_type = st.selectbox(
        "Tipo de alojamiento", np.insert(df.room_type.unique(), 0, "Todas"), (0)
    )

    query = f"price.between{price_range} and minimum_nights<={num_nights} and number_of_reviews>={num_reviews}"

    if room_type != "Todas":
        query = f'{query} and room_type=="{room_type}"'

    st.map(
        df.query(query)[["latitude", "longitude"]].dropna(how="any"),
        zoom=10,
    )

    st.header("Análisis de los barrios de Madrid")

    st.markdown(
        "Empezando a profundizar a nivel de barrio, vamos a previsualizar cual es el precio medio por noche en cada uno de los diferentes barrios de Madrid:"
    )

    st.bar_chart(
        df.groupby("neighbourhood_group")["price"].mean().sort_values(ascending=False)
    )

    st.markdown(
        "Según observamos en la gráfica, en base a la media de precios, es posible que existan outliers debido al pico de precio en la zona de San Blas. Procedemos a hacer limpieza de outliers usando el Z-Score y visualizamos de nuevo:"
    )

    norm_df = df[(np.abs(stats.zscore(df.price)) < 3)]
    st.bar_chart(
        norm_df.groupby("neighbourhood_group")["price"]
        .mean()
        .sort_values(ascending=False)
    )

    st.markdown(
        "Ahora observamos una distribución de los datos más normalizada, donde estos resultados confirman el análisis realizada de forma visual en el mapa, donde vemos que las zonas más elevadas de precio para alojarse son San Blas y Tetúan. Vemos también que Usera está en los barrios con el precio medio más elevado junto con Vicálvaro. En el caso de Vicalvaro se debe a la escasa oferta en la zona con precios muy elevados. Sin embargo en el caso de Usera, es una zona con mucha variedad en los rangos de precios."
    )

    st.markdown(
        "Entrando más a detalle, vamos a analizar la disponibilidad de alquileres por barrio. Para ello, veremos en detalle la feature `availability_365`. Opcionalmente, podemos ver la disponibilidad por barrio y precio máximo por noche"
    )

    set_max_price = st.checkbox("Establecer precio máximo")

    if set_max_price:
        max_price = st.number_input(
            label="Precio máximo",
            value=int(df.price.mean()),
            step=5,
        )
        set_max_price = f" and price<{max_price}"
    else:
        set_max_price = ""

    neighborhood = st.selectbox("Barrios", df.neighbourhood_group.unique())
    st.table(
        df.query(
            f"""neighbourhood_group==@neighborhood{set_max_price}\
        and availability_365>0"""
        )
        .availability_365.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        .to_frame()
        .T
    )

    st.header("Análisis por tipo de alojamiento")

    st.markdown(
        "Ahora que ya tenemos un primer vistazo de los datos de los barrios, vamos a ver que tipo de alojamiento son los que predominan en Madrid:"
    )

    st.bar_chart(data=df.room_type.value_counts())

    st.markdown(
        "Como podemos observar, predominan dos tipo de alojamientos: alojamientos enteros y habitaciones privadas. Los hoteles parece que no tienen cabida dentro de Airbnb (lógico) y las estancias privadas tampoco."
    )

    st.markdown("¿Pero, cómo se distribuye por barrio?")

    district = st.selectbox("Porcentaje de barrio", df.neighbourhood_group.unique())

    room_types_df = (
        df.groupby(["neighbourhood_group", "room_type"])
        .size()
        .reset_index(name="district_quantity")
    )
    st.table(room_types_df.loc[room_types_df["neighbourhood_group"] == district])

    st.markdown(
        """
    Con estos datos podemos ver el tipo de alojamiento que predomina por cada barrio. Si nos centramos en los barrios donde al principio vimos que predominaban precio bajos, como por ejemplo Usera, vemos que el alojamiento prediminante son las habitaciones privadas, superando por el doble de oferta al alojamiento entero.
    
    Sin embargo, si cambiamos a Tetúan, uno de los barrios donde se incrementaba la oferta según se incrementaba el rango de precio, vemos que la tendencia es justo la opuesta, predominan alojamientos enteros casi por el doble a habitaciones privadas. De esta forma, podemos establecer la hipótesis de que el tipo de alojamiento y su oferta en un barrio impacta directamente en el precio de los alojamientos en esa zona.
    """
    )

    st.markdown(
        "Siguiendo analizando los datos, vamos a ver cual es el precio medio por tipo de alojamiento:"
    )

    avg_price_room = (
        df.groupby("room_type")
        .price.mean()
        .reset_index()
        .round(2)
        .sort_values("price", ascending=False)
        .assign(avg_price=lambda x: x.pop("price").apply(lambda y: "%.1f" % y))
    )

    avg_price_room = avg_price_room.rename(
        columns={
            "room_type": "Tipo de alojamiento",
            "avg_price": "Precio medio ($)",
        }
    )

    st.table(avg_price_room)

    st.markdown(
        "Como se esperaba, el precio medio de los alojamientos enteros es superior a las habitaciones, haciendo que en barrios donde predomina la oferta de alojamientos enteros el precio de la zona sea más elevado. Por otro lado, también vemos una de las principales ventajas competitivas de Airbnb y su extendido uso frente a hoteles, el precio."
    )

    st.header("One more little thing...")

    st.markdown(
        f"De las {df.shape[1]} columnas disponibles en los datos, no todas aportan valor cualitativo e interés en nuestro análisis. En apartados posteriores profundizaremos en Feature Selection de cara al entrenamiento del modelo de predicción de precio."
    )

    st.markdown(
        "Disclaimer: de momento la forma más conveniente de filtrar los datos sería a través de las siguientes features: `price`, `room_type`, `minimum_of_nights`, `neighbourhood`, `name` y `number_of_reviews`"
    )

    defaultcols = [
        "price",
        "minimum_nights",
        "room_type",
        "neighbourhood",
        "name",
        "number_of_reviews",
    ]
    cols = st.multiselect("", df.columns.tolist(), default=defaultcols)
    st.dataframe(df[cols].head(10))

    st.header("Conclusiones")

    st.markdown(
        """
        - El precio de los alojamientos está directamente relacionado al tipo de alojamiento.
        - El precio es impactado en base a la oferta de tipo de alojamiento de cada barrio de Madrid.
        - La localización, cercanía y medios de transporte públicos cercanos tienen impacto en el precio.

        Me hubiera gustado profundizar más en los datos para extraer las siguientes conclusiones:
        - Qué barrios han crecido más a lo largo del tiempo basándonos en el incremento de listados por año.
        - Qué barrios han crecido más a lo largo del tiempo viendo la distribución de listados a lo largo del tiempo y los incrementos de precios.
        - Estudio de la temporalidad de los listados de apartamentos, cuales serían las épocas con más movimiento.
        """
    )


if __name__ == "__main__":
    write()
