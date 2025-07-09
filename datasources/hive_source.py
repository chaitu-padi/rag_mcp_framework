import pandas as pd
from pyhive import hive

from .base import DataSource


class HiveDataSource(DataSource):
    def __init__(self, host, port, username, database, query):
        self.host = host
        self.port = port
        self.username = username
        self.database = database
        self.query = query

    def load(self):
        conn = hive.Connection(
            host=self.host,
            port=self.port,
            username=self.username,
            database=self.database,
        )
        df = pd.read_sql(self.query, conn)
        return df.astype(str).apply(lambda row: " ".join(row), axis=1).tolist()
