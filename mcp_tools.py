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
    dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
    conn = cx_Oracle.connect(user, password, dsn)
    if query:
        sql = query
    elif table:
        sql = f"SELECT * FROM {table} FETCH FIRST {fetch_size} ROWS ONLY"
    else:
        raise ValueError("Either 'query' or 'table' must be provided for Oracle MCP context retrieval.")
    df = pd.read_sql(sql, conn)
    conn.close()
    # Convert to a readable string for LLM context
    if df.empty:
        return "[No data retrieved from Oracle]"
    # Show up to 5 rows as context
    preview = df.head(min(2, len(df))).to_string(index=False)
    return f"Oracle context (showing up to 2 rows):\n{preview}"


mcp_registry = MCPToolRegistry()
mcp_registry.register(
    "oracle_query",
    oracle_query_tool,
    description="Query Oracle DB using Model Context Protocol",
)
