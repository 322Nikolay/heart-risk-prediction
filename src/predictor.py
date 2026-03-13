import pandas as pd

from src.data_processing import DataProcessor
from src.model import HeartModel


class Predictor:

    def __init__(self, model_path, features_path):

        self.processor = DataProcessor()
        self.model = HeartModel(model_path, features_path)

    def predict_from_csv(self, csv_path):

        df = self.processor.load_data(csv_path)

        ids = df["id"]

        X = self.processor.preprocess(df)

        preds = self.model.predict(X)

        result = pd.DataFrame({
            "id": ids,
            "prediction": preds
        })

        return result