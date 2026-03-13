import pandas as pd


class DataProcessor:

    def __init__(self):
        pass

    def load_data(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:

        # удаляем id
        if "id" in df.columns:
            df = df.drop(columns=["id"])

        # кодируем gender
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].map({
                "Male": 1,
                "Female": 0
            })

        # заполняем пропуски
        df = df.fillna(0)

        return df