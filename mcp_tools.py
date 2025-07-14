# mcp_tools.py
"""
Integration layer for Model Context Protocol (MCP) tools.
This module provides a registry and interface for MCP tools that can be called from the RAG pipeline or prompt templates.
"""


class MCPToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name, func, description=None):
        self.tools[name] = {"func": func, "description": description or ""}

    def call(self, name, *args, **kwargs):
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not registered.")
        return self.tools[name]["func"](*args, **kwargs)

    def list_tools(self):
        return list(self.tools.keys())


# Register Oracle query tool for MCP
import cx_Oracle
import pandas as pd



def oracle_query_tool(user, password, host, port, service_name, table=None, query=None, fetch_size=100, **kwargs):
    """
    Pulls context from Oracle. If 'query' is provided, executes it. Otherwise, pulls all rows from 'table'.
    Returns a string summary of the result for LLM context.
    """
    print(f"[MCP] Connecting to Oracle with user={user}, password={'***' if password else '[EMPTY]'}, host={host}, port={port}, service_name={service_name}")
    print(f"[MCP] Table: {table}, Query: {query}, Fetch Size: {fetch_size}")
    import os
    # Print environment variable for password if PULL FROM ENV
    if isinstance(password, str):
        print(f"[MCP] Password value before env check: {password}")
        if password.upper().startswith("PULL FROM ENV:"):
            env_var = password.split(":", 1)[1].strip()
            env_pw = os.environ.get(env_var, None)
            print(f"[MCP] Password is set to PULL FROM ENV:{env_var}, resolved value: {env_pw}")
            password = env_pw
    dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
    try:
        conn = cx_Oracle.connect(user, password, dsn)
        print("[MCP] Oracle connection established.")
    except Exception as e:
        print(f"[MCP] Oracle connection failed: {e}")
        return f"[Oracle connection failed: {e}]"
    if query:
        sql = query
    elif table:
        sql = f"SELECT * FROM {table} FETCH FIRST {fetch_size} ROWS ONLY"
    else:
        print("[MCP] No query or table provided for Oracle context retrieval.")
        return "[No query or table provided for Oracle context retrieval]"
    print(f"[MCP] Executing SQL: {sql}")
    try:
        df = pd.read_sql(sql, conn)
        print(f"[MCP] Query executed. Rows returned: {len(df)}")
    except Exception as e:
        print(f"[MCP] SQL execution failed: {e}")
        conn.close()
        return f"[SQL execution failed: {e}]"
    conn.close()
    # Convert to a readable string for LLM context
    if df.empty:
        print("[MCP] No data retrieved from Oracle.")
        return "[No data retrieved from Oracle]"
    # Show up to 5 rows as context
    preview = df.head(min(2, len(df))).to_string(index=False)
    print(f"[MCP] Oracle context preview:\n{preview}")
    return f"Oracle context (showing up to 2 rows):\n{preview}"


mcp_registry = MCPToolRegistry()
mcp_registry.register(
    "oracle_query",
    oracle_query_tool,
    description="Query Oracle DB using Model Context Protocol",
)
