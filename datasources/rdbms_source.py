import pandas as pd
import sqlalchemy

from .base import DataSource


class RDBMSDataSource(DataSource):
    def __init__(self, connection_string, query):
        self.connection_string = connection_string
        self.query = query

    def load(self):
        engine = sqlalchemy.create_engine(self.connection_string)
        df = pd.read_sql(self.query, engine)
        return df.astype(str).apply(lambda row: " ".join(row), axis=1).tolist()
