import warnings

import pandas as pd
import seaborn as sns
import keras_tuner as kt
import streamlit as st

sns.set()
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

from geopy.geocoders import Nominatim
from geopy.distance import geodesic

from src.utils import get_data


@st.cache(allow_output_mutation=True)
class PriceModel:

    hot_encoder = None
    feature_selector = None
    model = None

    def _calculate_distance(self, longitude, latitude):
        madrid = "40.416729, -3.703339"
        return geodesic(madrid, f"{longitude} {latitude}").km

    def _replace_missing_data(self, data):
        missing_df = data.isnull().sum()
        missing_df = missing_df[missing_df > 0].sort_values(ascending=False)
        nan_columns = list(missing_df.to_dict().keys())

        for col in nan_columns:
            data_type = data[col].dtype
            if data_type == "object":
                data[col].fillna("NA", inplace=True)
            else:
                data[col].fillna(data[col].mean(), inplace=True)
        return data

    def _preprocess_train(self, data, predict=False):

        y = data["price"]

        data = data.drop(
            [
                "price",
                "name",
                "host_name",
                "host_id",
                "id",
                "number_of_reviews",
                "last_review",
                "reviews_per_month",
                "calculated_host_listings_count",
            ],
            axis=1,
        )

        data = self._replace_missing_data(data)

        data["distance"] = data.apply(
            lambda row: self._calculate_distance(row["longitude"], row["latitude"]),
            axis=1,
        )

        X = data.drop(["longitude", "latitude"], axis=1)

        self.hot_encoder = OneHotEncoder(handle_unknown="ignore")
        X = self.hot_encoder.fit_transform(X)

        self.feature_selector = SelectKBest(chi2, k=2)
        X = self.feature_selector.fit_transform(X, y)

        X = pd.DataFrame(X.toarray())

        return X, y

    def _preprocess_predict(self, data):

        data = self._replace_missing_data(data)

        data["distance"] = data.apply(
            lambda row: self._calculate_distance(row["longitude"], row["latitude"]),
            axis=1,
        )

        X = data.drop(["longitude", "latitude"], axis=1)
        X = self.hot_encoder.transform(X)
        X = self.feature_selector.transform(X)

        return pd.DataFrame(X.toarray())

    def _build_model(self, hp):
        model = Sequential()
        for i in range(hp.Int("layers", 2, 10)):
            model.add(
                Dense(
                    units=hp.Int(
                        "units_" + str(i), min_value=32, max_value=512, step=32
                    ),
                    activation="relu",
                )
            )
        model.add(Dense(1))
        model.compile(
            optimizer=Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
            loss="mse",
            metrics=["mse"],
        )
        return model

    def train_tuner(self, data):

        X, y = self._preprocess_train(data)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1
        )

        tuner = kt.RandomSearch(
            self._build_model,
            objective="val_mse",
            max_trials=1,
            executions_per_trial=3,
            directory="model_dir",
            project_name="House_Price_Prediction",
        )

        tuner.search(
            X_train,
            y_train,
            batch_size=128,
            epochs=200,
            validation_data=(X_test, y_test),
        )
        self.model = tuner.get_best_models(1)[0]

    def train_lr(self, data):
        X, y = self._preprocess_train(data)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1
        )

        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)

    def train(self, data, epochs=10, batch_size=64):
        X, y = self._preprocess_train(data)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1
        )

        # create model
        model = Sequential()
        model.add(Dense(32, input_dim=X_train.shape[1], activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2, input_shape=(32,)))
        model.add(Dense(64, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2, input_shape=(64,)))
        model.add(Dense(128, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2, input_shape=(128,)))
        model.add(Dense(64, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2, input_shape=(64,)))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1))
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.0001), loss="mse", metrics=["accuracy"]
        )

        model.summary()

        early_stop = EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=10
        )
        history = model.fit(
            x=X_train,
            y=y_train,
            validation_split=0.1,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stop],
        )
        self.model = model
        return model.history

    def predict(self, data):
        X = self._preprocess_predict(data)
        return self.model.predict(X)


def write():
    st.title("Price prediction Model Training")

    st.markdown(
        "Ha llegado el momento! Ahora vamos a realizar el entrenamiento de un modelo de Machine Learning para la predicción de precio de listado de una propiedad en Airbnb"
    )

    st.header("Premisas")

    st.markdown(
        """
    Este apartado se ha desarrollado teniendo en cuenta que este modelo será consumido por usuarios finales donde se intenta aportar valor. Por ello, se eleminarán todas las columnas que para el usuario no es intuitivo proporcionar y trabajaremos sobre ellas.

    Se han eliminado las siguientes columnas:
    - `price`
    - `name`
    - `host_name`
    - `host_id`
    - `id`
    - `number_of_reviews`
    - `last_review`
    - `reviews_per_month`
    - `calculated_host_listings_count`

    Trabajaremos utilizando de caja sólo estas columnas:
    - `neighbourhood_group`
    - `neighbourhood`
    - `latitude`
    - `longitude`
    - `room_type`
    - `price`
    - `minimum_nights`
    - `availability_365`
    """
    )

    st.header("Limpieza y selección de features")

    st.markdown(
        """
    Empezando por la limpieza, se han aplicado técnicas de limpieza estándar de datos:
    - Reemplazo de valores NaN por la media en caso de columnas numéricas y "NA" en caso de tipo string.
    - Aplicamos One Hot Encoding para las features categóricas para facilitar el posterior aprendizaje.
    """
    )

    st.markdown(
        """
    Por otro lado, se ha aplicado un procesamiento especial para las features de geolocalización `latitude` y `logitude`. Para extraer valor de estas features y que tengan un valor relevante respecto al resto de datos, se procesarán ambas columnas para generar una nueva feature que llamaremos `distance`. 
    Esta feature es el resultado del cálculo de la distancia en Km de la posición que representa cada entrada y una posición de referencia, en este caso la Puerta del Sol de Madrid.
    """
    )

    st.markdown(
        "Posteriormente, una vez finalizado todo el preprocesamiento se realiza Univariate Feature Selection usando k highest scores usando chi2"
    )

    st.markdown(
        "Una vez tenemos todo el procesamiento completado, dividimos en splits de entrenamiento y test los datos. Para esta separación de los datos usaremos un 80% entrenamiento y 20% para test. La cantidad de datos no es muy grande por lo que usamos esa distribución. Si tuvieramos millones de samples optaríamos por una distribución más generosa como 98/2."
    )

    st.markdown(
        "Respecto al modelo en si, se podría haber optado por un enfoque conservador usando Logistic Regression o Gradient Boosting pero he optado una Neural Network con Tensorflow. Seguramente, dado la cantidad de datos la diferencia de performance entre los diferentes métodos no sea destacable y el hecho de usar una red neuronal me la juego a sufrir un poco de overfitting sobre los datos. Nada que no se pueda resolver con una dosis de Dropout o Regularización entre las capas."
    )

    st.markdown(
        "En cuanto a la selección de la arquitectura del modelo e hiperparámeteros he implementado un tuner usar Random Search para que optimice parámetros así como número de neuronas de la red automáticamente. La ejecución con esta metodología puede alargarse bastante en el tiempo, asique he desarrollado yo una propia en base a mi criterio para evitar que quien esté leyendo esto pierda más el tiempo."
    )

    st.markdown(
        "Redoble de tambores... ¡Ya estamos listos para empezar! Tan solo haz click en el siguiente botón para realizar el entrenamiento del modelo y... que empiece lo bueno :)"
    )

    data = get_data()
    model = PriceModel()

    st.markdown(
        "Se puede personalizar la ejecución para que dure menos o más epochs al gusto. Aún así el entrenamiento tiene Early Stopping en el callback de entrenamiento."
    )

    if st.button("Lanzar entrenamiento con Logistic Regression!"):
        with st.spinner("Haciendo huevos fritos sobre el ordenador..."):
            model.train_lr(data)
        st.success("Terminado!")

    st.number_input("epochs", min_value=1)
    st.number_input("batch_size", min_value=32)

    if st.button("Lanzar entrenamiento con Neural Network!"):
        with st.spinner("Haciendo huevos fritos sobre el ordenador..."):
            model.train(data)
        st.success("Terminado!")

    st.header("Let's predict!")

    st.markdown(
        "Una vez ya tenemos nuestro modelo entrenado, vamos a probar que es capaz de hacer una inferencia y darnos una estimación de precio para un listado. Utilizaremos el siguiente ejemplo:"
    )
    st.code(
        """
testing = pd.DataFrame.from_dict([{"neighbourhood_group": "Chamartín", "neighbourhood": "Hispanoamérica", "latitude": 40.418595550000006, "longitude": -3.702305961540991, "room_type": "Private room", "minimum_nights": 1, "availability_365": 82}])
model.predict(testing)
        """
    )

    if st.button("Lanzar inferencia de prueba"):
        with st.spinner("Viendo el futuro..."):
            testing = pd.DataFrame.from_dict(
                [
                    {
                        "neighbourhood_group": "Chamartín",
                        "neighbourhood": "Hispanoamérica",
                        "latitude": 40.418595550000006,
                        "longitude": -3.702305961540991,
                        "room_type": "Private room",
                        "minimum_nights": 1,
                        "availability_365": 82,
                    }
                ]
            )
            result = model.predict(testing)
        st.success(f"El precio por noche recomendado sería: __{float(result)}__")


if __name__ == "__main__":
    write()
