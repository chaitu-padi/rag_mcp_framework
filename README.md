
# Modular RAG Framework (MCP-Ready)

This project is a modular Retrieval-Augmented Generation (RAG) framework designed for extensibility, configurability, and robust workflow management. It supports multiple data sources (including Oracle/MCP), vector databases, embedding models, and LLMs. The framework is suitable for both experimentation and production use.

---

## Key Features

- **Modular architecture**: Easily add new data sources, embedding models, LLMs, or vector DBs by implementing the appropriate base class and registering in the UI.
- **Batch embedding**: Efficient, configurable batch processing for embeddings with batch size control.
- **Oracle/MCP support**: Native integration for Oracle and Model Context Protocol (MCP) context retrieval, with fallback to vector DB.
- **Streamlit UI**: User-friendly interface for workflow management, configuration, and running RAG pipelines. UI supports:
  - Viewing, creating, and editing workflows
  - Saving/loading workflows by unique workflow ID
  - Editing all pipeline parameters (data source, vector DB, embedding, LLM, prompt variables)
  - Running data ingestion, LLM inference, or both together
  - Viewing all existing workflows
- **Workflow management**: Save, load, and list workflows by unique workflow ID. Workflows are stored as YAML configs under `config/`.
- **Configurable via YAML/UI**: All parameters are grouped under `data_ingestion` and `llm_inference` in the config. The UI and YAML are always in sync.
- **Separation of concerns**: Decoupled scripts for data ingestion (`data_ingest.py`) and LLM inference (`llm_infer.py`).
- **Logging and documentation**: Clear logs and code comments for maintainability.

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
    params: {...}
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
```

Workflows can be saved/loaded by unique workflow ID. All configuration can be managed via the UI or YAML. The UI always reflects the current config and allows editing all parameters.

---

## Usage

### Streamlit UI

Run the UI:

```sh
streamlit run app.py
```

Features:
- View, create, and edit workflows
- Save/load workflows by ID
- Configure all pipeline parameters (data source, vector DB, embedding, LLM, prompt variables)
- Run data ingestion, LLM inference, or both together
- View all existing workflows

### CLI Scripts

For automation or batch runs:

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

**Cons:**
- MCP/Oracle setup may require additional infrastructure
- May be slower than pure vector DB for some queries

---

## License

MIT

---

## Pros and Cons of Adding MCP (Model Context Protocol) in RAG

### Pros
- **Real-Time Data Retrieval:**  
  MCP enables the RAG pipeline to fetch the most up-to-date context directly from the source (e.g., Oracle DB) at query time, ensuring answers are always based on the latest data.
- **Scalability:**  
  By retrieving only the relevant context for each query, MCP avoids the need to embed and store the entire dataset, making the system more scalable for large and dynamic data sources.
- **Reduced Maintenance Overhead:**  
  No need to re-embed or re-index the entire dataset when the source data changes, as MCP always queries live data.
- **Extensibility:**  
  MCP allows easy integration of new data sources and custom retrieval logic by defining new MCP tools, making the framework adaptable to various enterprise needs.
- **Configurable and Modular:**  
  MCP tools and parameters are fully configurable via YAML, supporting flexible workflows and rapid experimentation.

### Cons
- **Potential Latency:**  
  Querying live data sources (like databases) at inference time can introduce additional latency compared to retrieving pre-embedded vectors from a vector DB.
- **Complexity:**  
  Implementing and maintaining MCP tools for different data sources adds complexity to the codebase and configuration.
- **Dependency on Source Availability:**  
  The RAG pipeline’s ability to answer questions depends on the availability and performance of the underlying data source (e.g., Oracle DB uptime).
- **Limited by Source Query Performance:**  
  The speed and efficiency of context retrieval are constrained by the performance of the source system and the efficiency of the queries defined in MCP tools.
- **Security and Access Control:**  
  Direct database access at inference time may require careful management of credentials, permissions, and audit trails.

---

**Summary:**  
MCP in RAG offers powerful real-time, scalable, and extensible context retrieval, but introduces trade-offs in latency, complexity, and dependency on source systems. It is best suited for scenarios where data freshness and flexibility outweigh the need for ultra-low-latency responses.

---

## Troubleshooting
- Ensure all required API keys are set (either in the UI, config, or as environment variables).
- If you encounter version or schema errors with ChromaDB or NumPy, follow the compatibility instructions in `requirements.txt`.
- For large datasets, consider context window management or chunking strategies (see `rag/rag_pipeline.py`).
- If you add new data sources or embedding types, ensure they are imported and handled in `main.py`.

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
