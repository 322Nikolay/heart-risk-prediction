from fastapi import FastAPI
from src.predictor import Predictor

app = FastAPI()

predictor = Predictor(
    "models/model.pkl",
    "models/features.pkl"
)


@app.post("/predict")
def predict(csv_path: str):

    preds = predictor.predict_from_csv(csv_path)

    return preds.to_dict(orient="records")