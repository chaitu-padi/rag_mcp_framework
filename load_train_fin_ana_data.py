import os
import cx_Oracle
import pandas as pd

def load_train_fin_ana_data(xls_path, table_name, dsn, user, password, if_exists="replace"):
    """
    Load data from train-FIN_ANA_DATA.xls into a new Oracle table.
    :param xls_path: Path to the Excel file (train-FIN_ANA_DATA.xls)
    :param table_name: Target Oracle table name
    :param dsn: Oracle DSN string (use cx_Oracle.makedsn)
    :param user: Oracle username
    :param password: Oracle password
    :param if_exists: 'replace' to drop and recreate table, 'append' to insert
    """
    df = pd.read_excel(xls_path)
    print(f"Loaded {len(df)} rows from {xls_path}")

    conn = cx_Oracle.connect(user, password, dsn)
    cursor = conn.cursor()

    if if_exists == "replace":
        try:
            cursor.execute(f"DROP TABLE {table_name}")
            print(f"Dropped table {table_name}")
        except cx_Oracle.DatabaseError as e:
            (error_obj,) = e.args
            if error_obj.code == 942:  # ORA-00942: table or view does not exist
                print(f"Table {table_name} does not exist. Skipping drop.")
            else:
                raise

    # Create table if not exists (schema inferred from DataFrame)
    columns = []
    for col, dtype in zip(df.columns, df.dtypes):
        col_clean = col.replace(' ', '_').replace('-', '_').upper()
        if pd.api.types.is_integer_dtype(dtype):
            columns.append(f'{col_clean} NUMBER')
        elif pd.api.types.is_float_dtype(dtype):
            columns.append(f'{col_clean} NUMBER')
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            columns.append(f'{col_clean} DATE')
        else:
            columns.append(f'{col_clean} VARCHAR2(255)')
    create_stmt = f"CREATE TABLE {table_name} (" + ", ".join(columns) + ")"
    try:
        cursor.execute(create_stmt)
        print(f"Created table {table_name}")
    except cx_Oracle.DatabaseError as e:
        (error_obj,) = e.args
        if error_obj.code == 955:  # ORA-00955: name is already used by an existing object
            print(f"Table {table_name} already exists. Skipping creation.")
        else:
            raise

    # Insert data
    col_names = [col.replace(' ', '_').replace('-', '_').upper() for col in df.columns]
    placeholders = ", ".join([f":{i+1}" for i in range(len(col_names))])
    insert_stmt = f"INSERT INTO {table_name} (" + ", ".join(col_names) + f") VALUES ({placeholders})"
    # Convert all values to string or None to avoid type issues
    def clean_row(row):
        return tuple(str(val) if not pd.isna(val) else None for val in row)
    data = [clean_row(row) for row in df.itertuples(index=False, name=None)]
    try:
        cursor.executemany(insert_stmt, data)
        conn.commit()
        print(f"Inserted {len(data)} rows into {table_name}")
    except cx_Oracle.DatabaseError as e:
        print(f"Error inserting data: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    # Example usage
    dsn = cx_Oracle.makedsn("localhost", 1521, service_name="FREEPDB1")
    load_train_fin_ana_data(
        xls_path="train-FIN_ANA_DATA.xls",
        table_name="FIN_ANA_DATA",
        dsn=dsn,
        user="system",
        password="opass1231",
        if_exists="replace",  # or "append"
    )
