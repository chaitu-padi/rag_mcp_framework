import cx_Oracle
import pandas as pd

from .base import DataSource


class OracleDataSource(DataSource):
    def __init__(
        self,
        user,
        password,
        host,
        port,
        service_name,
        table,
        query=None,
        fetch_size=100,
    ):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.service_name = service_name
        self.table = table
        self.query = query or f"SELECT * FROM {table}"
        self.fetch_size = fetch_size

    def load(self):
        dsn = cx_Oracle.makedsn(self.host, self.port, service_name=self.service_name)
        conn = cx_Oracle.connect(self.user, self.password, dsn)
        df = pd.read_sql(self.query, conn)
        conn.close()
        return df.astype(str).apply(lambda row: " ".join(row), axis=1).tolist()
