# Modular RAG Framework (MCP-Ready)

This project is a modular Retrieval-Augmented Generation (RAG) framework designed for extensibility, configurability, and robust workflow management. It supports multiple data sources (including Oracle/MCP), vector databases, embedding models, and LLMs. The framework is suitable for both experimentation and production use.

---

## Key Features

- **Modular architecture**: Easily add new data sources, embedding models, LLMs, or vector DBs by implementing the appropriate base class and registering in the UI.
- **Batch embedding**: Efficient, configurable batch processing for embeddings with batch size control.
- **Oracle/MCP support**: Native integration for Oracle and Model Context Protocol (MCP) context retrieval, with fallback to vector DB. **Oracle credentials can be securely managed via environment variables.**
- **Streamlit UI**: User-friendly interface for workflow management, configuration, and running RAG pipelines. UI supports:
  - Viewing, creating, and editing workflows
  - Saving/loading workflows by unique workflow ID
  - Editing all pipeline parameters (data source, vector DB, embedding, LLM, prompt variables)
  - Secure handling of Oracle passwords: password is never shown in the UI, and can be resolved from environment variables for ingestion and inference
  - Running data ingestion, LLM inference, or both together
  - Viewing all existing workflows
  - Viewing execution logs in a dedicated right-side panel
- **Workflow management**: Save, load, and list workflows by unique workflow ID. Workflows are stored as YAML configs under `config/`.
- **Configurable via YAML/UI**: All parameters are grouped under `data_ingestion` and `llm_inference` in the config. The UI and YAML are always in sync.
- **Separation of concerns**: Decoupled scripts for data ingestion (`data_ingest.py`) and LLM inference (`llm_infer.py`).
- **Logging and documentation**: Clear logs and code comments for maintainability. All pipeline execution logs are shown in the UI for debugging.
- **Environment variable support**: API keys and Oracle passwords can be set as environment variables and referenced in the config using `PULL FROM ENV:VARNAME`.

---

## Project Structure

```
app.py                # Streamlit UI for workflow management and running RAG
data_ingest.py        # Script for data ingestion (data source -> vector DB)
llm_infer.py          # Script for LLM inference (querying the vector DB and LLM)
main.py               # Deprecated (see new scripts above)
config/
  config.yaml         # Main configuration file (split into data_ingestion and llm_inference)
  workflow_utils.py   # Utilities for workflow save/load/list
datasources/          # Modular data source handlers (CSV, RDBMS, Hive, Oracle, etc.)
embeddings/           # Embedding model handlers (OpenAI, SentenceTransformers, etc.)
llms/                 # LLM handlers (OpenAI, etc.)
rag/                  # RAG pipeline logic
vectordb/             # Vector DB handlers (ChromaDB, etc.)
README.md             # This file
...                   # Other supporting files
```

---

## Configuration

Configuration is managed via YAML and the Streamlit UI. The config is split into two main sections:

```
data_ingestion:
  datasource:
    type: CSV | RDBMS | Hive | Oracle
    params:
      user: ...
      password: ... # Can be 'PULL FROM ENV:VARNAME' for secure env var usage
  vectordb:
    type: ChromaDB
    params: {...}
  embedding:
    type: OpenAI | SentenceTransformers
    params: {...}
llm_inference:
  llm:
    type: OpenAI
    params: {...}
  prompt_vars:
    ...
  mcp:
    enabled: true
    fallback_to_vectordb: false
    tools:
      - name: oracle_query
        datasource: oracle
        params:
          user: ...
          password: ... # Can be 'PULL FROM ENV:VARNAME'
          ...existing code...
  query:
    top_k: 10
```

Workflows can be saved/loaded by unique workflow ID. All configuration can be managed via the UI or YAML. The UI always reflects the current config and allows editing all parameters. **Sensitive fields like Oracle password are never shown in the UI, but are resolved from environment variables when running the pipeline.**

---

## Usage

### Streamlit UI (Work in Progress)

Run the UI:

```sh
streamlit run app.py
```

Features:
- View, create, and edit workflows
- Save/load workflows by ID
- Configure all pipeline parameters (data source, vector DB, embedding, LLM, prompt variables)
- Securely manage Oracle credentials via environment variables
- Run data ingestion, LLM inference, or both together
- View all existing workflows
- View execution logs in a dedicated right-side panel for debugging

### CLI Scripts

For automation or batch runs:

For testing, run as below : By default it will take the config.yaml
```sh
python data_ingest.py 
python llm_infer.py 
```

For future state run : Configurations will be taken from workflow_id.yaml
```sh
python data_ingest.py --workflow_id <id>
python llm_infer.py --workflow_id <id> --question "..."
```

---

## Extensibility

Add new data sources, embedding models, LLMs, or vector DBs by implementing the appropriate base class in their respective folders and registering them in `app.py`.

---

## MCP in RAG: Pros & Cons

**Pros:**
- Direct context retrieval from Oracle/MCP for high accuracy
- Fallback to vector DB for flexibility
- Secure credential management via environment variables

**Cons:**
- MCP/Oracle setup may require additional infrastructure
- May be slower than pure vector DB for some queries
- Requires careful credential management

---

## License

MIT

---

## Troubleshooting
- Ensure all required API keys and Oracle passwords are set (either in the UI, config, or as environment variables).
- If you encounter version or schema errors with ChromaDB or NumPy, follow the compatibility instructions in `requirements.txt`.
- For large datasets, consider context window management or chunking strategies (see `rag/rag_pipeline.py`).
- If you add new data sources or embedding types, ensure they are imported and handled in `app.py`.
- If Oracle connection fails, check that the password is set in the environment and referenced as `PULL FROM ENV:VARNAME` in the config.

---

## Directory Structure
```
├── app.py                # Streamlit UI
├── data_ingest.py        # CLI entry point for data ingestion
├── llm_infer.py          # CLI entry point for llm inference
├── config/
│   └── config.yaml       # Configuration file
├── datasources/          # Data source modules
├── embeddings/           # Embedding model modules
├── vectordb/             # Vector DB modules
├── llms/                 # LLM modules
├── rag/
│   └── rag_pipeline.py   # RAG pipeline logic
├── mcp_tools.py          # MCP tool registry and integration
├── requirements.txt      # Python dependencies
└── ...
```

---

## Contact
For questions or contributions, please refer to the project repository or contact the maintainer.
