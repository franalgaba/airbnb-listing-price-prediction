import warnings

import pandas as pd
import seaborn as sns
import keras_tuner as kt
import streamlit as st

sns.set()
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")


from src.utils import get_data, calculate_distance


@st.cache(allow_output_mutation=True)
class PriceModel:

    """
    Class for listing price modeling and processing
    """

    hot_encoder = None
    feature_selector = None
    model = None

    def _replace_missing_data(self, data):
        """
        Method for replacing NaN values for mean in numerics and NA in string
        """

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

        """
        Preprocess train data and prepares it for training.
        """

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
            lambda row: calculate_distance(row["longitude"], row["latitude"]),
            axis=1,
        )

        X = data.drop(["longitude", "latitude"], axis=1)

        self.hot_encoder = OneHotEncoder(handle_unknown="ignore")
        X = self.hot_encoder.fit_transform(X)

        # self.feature_selector = SelectKBest(chi2, k=2)
        # X = self.feature_selector.fit_transform(X, y)

        X = pd.DataFrame(X.toarray())

        return X, y

    def _preprocess_predict(self, data):

        """
        Preprocess data that is going to be consumed on inference
        """

        data = self._replace_missing_data(data)

        data["distance"] = data.apply(
            lambda row: calculate_distance(row["longitude"], row["latitude"]),
            axis=1,
        )

        X = data.drop(["longitude", "latitude"], axis=1)
        X = self.hot_encoder.transform(X)
        # X = self.feature_selector.transform(X)

        return pd.DataFrame(X.toarray())

    def _build_model(self, hp):

        """
        Builds a Keras model using Keras Tuner for architecture search
        and hyperparameter tuning.
        """

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

        """
        Model tuning and hyperparameters fine tunning for training
        """

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

    def train_lr(self, data, max_iter=5):

        """
        Trains a Logistic Regression model and validates
        using cross-validation.
        """

        X, y = self._preprocess_train(data)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1
        )

        self.model_lr = LogisticRegression(max_iter=5)
        self.model_lr.fit(X_train, y_train)

        kfold = KFold(n_splits=10, random_state=7, shuffle=True)
        results = cross_val_score(self.model_lr, X, y, cv=kfold, scoring="roc_auc")
        st.markdown("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))

    def train(self, data, epochs=10, batch_size=64, learning_rate=0.0001):

        """
        Main training method for training a custom designed Keras Model
        """

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
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1))
        # Compile model
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

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

    def predict(self, data, is_lr=False):

        """
        Main inference method using previously trained models
        """

        if self.hot_encoder is None:
            st.error(
                "Para hacer la inferencia previamente se tiene que haber ejecutado un entrenamiento"
            )
        else:
            X = self._preprocess_predict(data)
            result = None
            if is_lr:
                if self.model_lr is None:
                    st.error(
                        "Modelo de Logistic Regression no entrenado. Por favor, entr??nalo antes de ejecutar la predicci??n."
                    )
                else:
                    result = self.model_lr.predict(X)
            else:
                if self.model is None:
                    st.error(
                        "Modelo de Red Neuronal no entrenado. Por favor, entr??nalo antes de ejecutar la predicci??n."
                    )
                else:
                    result = self.model.predict(X)
            return result


def write():

    """
    Main entrypoint for the Model Training page
    """

    st.title("Price prediction Model Training")

    st.markdown(
        "Ha llegado el momento! Ahora vamos a realizar el entrenamiento de un modelo de Machine Learning para la predicci??n de precio de listado de una propiedad en Airbnb."
    )

    st.header("Premisas")

    st.markdown(
        """
    Este apartado se ha desarrollado teniendo en cuenta que este modelo ser?? consumido por usuarios finales donde se intenta aportar valor. Por ello, se eleminar??n todas las columnas que para el usuario no es intuitivo proporcionar y trabajaremos sobre ellas.

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

    Trabajaremos utilizando de caja s??lo estas columnas:
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

    st.header("Limpieza y selecci??n de features")

    st.markdown(
        """
    Empezando por la limpieza, se han aplicado t??cnicas de limpieza est??ndar de datos:
    - Reemplazo de valores NaN por la media en caso de columnas num??ricas y "NA" en caso de tipo string.
    - Aplicamos One Hot Encoding para las features categ??ricas para facilitar el posterior aprendizaje.
    """
    )

    st.markdown(
        """
    Por otro lado, se ha aplicado un procesamiento especial para las features de geolocalizaci??n `latitude` y `logitude`. Para extraer valor de estas features y que tengan un valor relevante respecto al resto de datos, se procesar??n ambas columnas para generar una nueva feature que llamaremos `distance`. 
    Esta feature es el resultado del c??lculo de la distancia en Km de la posici??n que representa cada entrada y una posici??n de referencia, en este caso la Puerta del Sol de Madrid.
    """
    )

    st.markdown(
        "Una vez tenemos todo el procesamiento completado, dividimos en splits de entrenamiento y test los datos. Para esta separaci??n de los datos usaremos un 80% entrenamiento y 20% para test. La cantidad de datos no es muy grande por lo que usamos esa distribuci??n. Si tuvieramos millones de samples optar??amos por una distribuci??n m??s generosa como 98/2."
    )

    st.markdown(
        "Respecto al modelo en si, se podr??a haber optado por un enfoque conservador usando Logistic Regression o Gradient Boosting pero he optado una Neural Network con Tensorflow. Seguramente, dado la cantidad de datos la diferencia de performance entre los diferentes m??todos no sea destacable y el hecho de usar una red neuronal me la juego a sufrir overfitting sobre los datos. Nada que no se pueda resolver con una dosis de Dropout o Regularizaci??n entre las capas, o no, veremos."
    )

    st.markdown(
        "En cuanto a la selecci??n de la arquitectura del modelo e hiperpar??meteros he implementado un tuner usando Random Search para que optimice par??metros as?? como n??mero de neuronas de la red autom??ticamente. La ejecuci??n con esta metodolog??a puede alargarse bastante en el tiempo, asique he desarrollado yo una propia en base a mi criterio para evitar que quien est?? leyendo esto pierda m??s el tiempo. A??n as?? la implementaci??n est?? [aqu??](https://github.com/franalgaba/airbnb-listing-price-prediction/blob/main/src/training.py#L139)"
    )

    st.markdown(
        "Redoble de tambores... ??Ya estamos listos para empezar! Tan solo haz click en el siguiente bot??n para realizar el entrenamiento del modelo y... ??que bailen esos pesos!"
    )

    data = get_data()
    model = PriceModel()

    st.markdown(
        "Se puede personalizar la ejecuci??n para que dure menos o m??s epochs al gusto. A??n as?? el entrenamiento tiene Early Stopping en el callback de entrenamiento."
    )

    max_iter = st.number_input("max_iter", min_value=5)

    st.warning(
        "Por favor, no lanzar entrenamientos desde alg??n sitio que no sea local. El servicio se quedar?? sin recursos y morir??. Para ejecutar en local consultar el [README](https://github.com/franalgaba/airbnb-listing-price-prediction/blob/main/README.md)"
    )
    if st.button("Lanzar entrenamiento con Logistic Regression!"):
        with st.spinner(":fire: Haciendo huevos fritos sobre el ordenador..."):
            model.train_lr(data, max_iter)
        st.success("Terminado!")

    st.markdown(
        """
    Los par??metros m??s ??ptimos encontrados son:
    - `epochs=100`
    - `batch_size=128`
    - `learning_rate=0.001`
    """
    )
    epochs = st.number_input("epochs", min_value=1)
    batch_size = st.number_input("batch_size", min_value=32)
    learning_rate = st.number_input("learning_rate", min_value=0.0001, format="%.4f")

    st.warning(
        "Por favor, no lanzar entrenamientos desde alg??n sitio que no sea local. El servicio se quedar?? sin recursos y morir??. Para ejecutar en local consultar el [README](https://github.com/franalgaba/airbnb-listing-price-prediction/blob/main/README.md)"
    )
    if st.button("Lanzar entrenamiento con Neural Network!"):
        with st.spinner(":fire: Haciendo huevos fritos sobre el ordenador..."):
            history = model.train(data, epochs, batch_size, learning_rate)
            st.line_chart(history.history)
        st.success("Terminado!")
        st.markdown(
            "Como vemos en la gr??fica superior se produce mucho overfitting que no es f??cil de controlar debido a la poca cantidad de datos. Las NN no son una soluci??n adecuada al problema."
        )

    st.header("Let's predict!")

    st.markdown(
        "Una vez ya tenemos nuestro modelo entrenado, vamos a probar que es capaz de hacer una inferencia y darnos una estimaci??n de precio para un listado. Utilizaremos el siguiente ejemplo:"
    )
    st.code(
        """
testing = pd.DataFrame.from_dict([{"neighbourhood_group": "Chamart??n", "neighbourhood": "Hispanoam??rica", "latitude": 40.418595550000006, "longitude": -3.702305961540991, "room_type": "Private room", "minimum_nights": 1, "availability_365": 82}])
model.predict(testing)
        """
    )

    selected_model = st.selectbox(
        "Modelo de predicc??n", ("Logistic Regression", "Neural Network")
    )

    if st.button("Lanzar inferencia de prueba"):
        with st.spinner("Viendo el futuro..."):
            testing = pd.DataFrame.from_dict(
                [
                    {
                        "neighbourhood_group": "Chamart??n",
                        "neighbourhood": "Hispanoam??rica",
                        "latitude": 40.418595550000006,
                        "longitude": -3.702305961540991,
                        "room_type": "Private room",
                        "minimum_nights": 1,
                        "availability_365": 82,
                    }
                ]
            )
            is_lr = False
            if selected_model == "Logistic Regression":
                is_lr = True
            result = model.predict(testing, is_lr)
        if result is not None:
            st.success(f"El precio por noche recomendado ser??a: __{float(result)}__")


if __name__ == "__main__":
    write()
