import os
import time
import tempfile

import pickle
import tensorflow as tf
from loguru import logger


from app.core.messages import NO_VALID_PAYLOAD
from app.models.payload import ListingPayload, payload_to_df
from app.models.prediction import ListingPredictionResult
from app.core.config import (
    KERAS_MODEL,
    HOT_ENCODER_MODEL,
)


class ListingPriceModel:
    def __init__(
        self,
        model_dir,
    ):
        self.model_dir = model_dir
        self._load_local_model()

    def _load_local_model(self):
        keras_model_path = os.path.join(self.model_dir, KERAS_MODEL)
        with tempfile.NamedTemporaryFile(suffix=".h5") as local_file:
            with tf.io.gfile.GFile(keras_model_path, mode="rb") as gcs_file:
                local_file.write(gcs_file.read())
                self.model = tf.keras.models.load_model(local_file.name, compile=False)

        hot_encoder_path = os.path.join(self.model_dir, HOT_ENCODER_MODEL)
        self.hot_encoder = pickle.load(tf.io.gfile.GFile(hot_encoder_path, mode="rb"))

    def _pre_process(self, payload: ListingPayload) -> str:
        logger.debug("Pre-processing payload.")
        return payload_to_df(payload)

    def _predict(self, data) -> float:
        logger.debug("Predicting.")
        # One Hot data
        x_test = self.hot_encoder.transform(data)
        # Predict
        return self.model.predict(x_test)[0]

    def _post_process(
        self, prediction: float, start_time: float
    ) -> ListingPredictionResult:
        logger.debug("Post-processing prediction.")

        return ListingPredictionResult(
            price=prediction,
            elapsed_time=(time.time() - start_time),
        )

    def predict(self, payload: ListingPayload):
        if payload is None:
            raise ValueError(NO_VALID_PAYLOAD.format(payload))

        start_at = time.time()
        pre_processed_payload = self._pre_process(payload)
        prediction = self._predict(pre_processed_payload)
        post_processed_result = self._post_process(prediction, start_at)

        return post_processed_result
