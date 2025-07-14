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
        print(f"[OracleDataSource] Connecting to Oracle with:")
        print(f"  user: '{self.user}'")
        print(f"  host: '{self.host}'")
        print(f"  port: '{self.port}'")
        print(f"  service_name: '{self.service_name}'")
        print(f"  table: '{self.table}'")
        print(f"  query: '{self.query}'")
        # Do NOT print password for security
        dsn = cx_Oracle.makedsn(self.host, self.port, service_name=self.service_name)
        try:
            conn = cx_Oracle.connect(self.user, self.password, dsn)
        except cx_Oracle.DatabaseError as e:
            print(f"[OracleDataSource] cx_Oracle.DatabaseError: {e}")
            print("[OracleDataSource] See https://docs.oracle.com/error-help/db/ora-01017/")
            raise
        df = pd.read_sql(self.query, conn)
        conn.close()
        return df.astype(str).apply(lambda row: " ".join(row), axis=1).tolist()
