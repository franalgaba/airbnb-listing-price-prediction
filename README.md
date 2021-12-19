# Airbnb Listing Price Analysis and Prediction

In this repository you will find an implementation for listing price of Airbnb homes using Inside Airbnb dataset. The project is structured in two parts:

- EDA and Model training using a Streamlit front.
- FastAPI microservice for serving inference in the front.

The Streamlit front is structured in the `src` folder with one script per navigation section:
- EDA: `src/eda.py`.
- Model Training: `src/training.py`.
- Model deployment: `src/price_predict.py`

The prediction microservice is structured in the `prediction-service/app` folder:
- API specification and implementation: `app/api`
- Core functionality: `src/core`
- Request schemas: `src/models`
- Inference implementation: `src/services`

## Instructions

For executing the Streamlit app:

1. Python 3.8+ installation.
2. Clone the repository: `git clone <repository>`.
3. Go the repository: `cd airbnb-listing-price-prediction`.
4. Create a venv: `python -m venv venv`.
5. Install dependencies: `pip install -r requirements.txt`.
6. Execute: `streamlit run app.py --server.headless true`.
7. Go to the URL Streamlit gives you.
8. Have fun!

For executing the FastAPI microservice:

1. Python 3.8+ installation.
2. Clone the repository: `git clone <repository>`.
3. Go the repository: `cd airbnb-listing-price-prediction/prediction-service`.
4. Create a venv: `python -m venv venv`.
5. Install dependencies: `pip install -r requirements.txt`.
6. Set environment variable: `export DEFAULT_MODEL_PATH="gs://keepler-inference-models/"`.
6. Execute: `uvicorn app.main:app`.
7. Go to: `http://127.0.0.1:8000`.
8. Have fun!
