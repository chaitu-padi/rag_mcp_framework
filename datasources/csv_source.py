import pandas as pd

from .base import DataSource


class CSVDataSource(DataSource):
    def __init__(self, path):
        self.path = path

    def load(self):
        df = pd.read_csv(self.path)
        # Return as list of strings (one per row)
        return df.astype(str).apply(lambda row: " ".join(row), axis=1).tolist()
