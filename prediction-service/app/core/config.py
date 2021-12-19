import os

APP_VERSION = "0.0.1"
APP_NAME = "Airbnb Listing Price Prediction"
API_PREFIX = "/api"

IS_DEBUG = os.getenv("IS_DEBUG", False)
DEFAULT_MODEL_PATH = os.getenv("DEFAULT_MODEL_PATH")

# EXPORT
KERAS_MODEL = "model.h5"
HOT_ENCODER_MODEL = "hot_encoder.pkl"