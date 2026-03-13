import joblib


class HeartModel:

    def __init__(self, model_path, features_path):

        self.model = joblib.load(model_path)
        self.features = joblib.load(features_path)

    def predict(self, X):

        # приводим к правильному порядку признаков
        X = X[self.features]

        probs = self.model.predict_proba(X)[:, 1]

        threshold = 0.52

        preds = (probs > threshold).astype(int)

        return preds