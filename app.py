import os
import streamlit as st
import yaml
from datasources.csv_source import CSVDataSource
from datasources.hive_source import HiveDataSource
from datasources.oracle_source import OracleDataSource
from datasources.rdbms_source import RDBMSDataSource
from embeddings.openai_embed import OpenAIEmbedding
from embeddings.sentence_transformers_embed import SentenceTransformersEmbedding
from llms.openai_llm import OpenAILLM
from rag.rag_pipeline import RAGPipeline
from vectordb.chromadb import ChromaDB
from config.workflow_utils import save_workflow, load_workflow, list_workflows

# Supported options
EMBEDDING_MODELS = {
    "OpenAI": OpenAIEmbedding,
    "SentenceTransformers": SentenceTransformersEmbedding,
}
LLM_MODELS = {
    "OpenAI": OpenAILLM,
    # Add more LLMs here
}
VECTOR_DBS = {
    "ChromaDB": ChromaDB,
    # Add more vector DBs here
}
DATASOURCES = {
    "CSV": CSVDataSource,
    "RDBMS": RDBMSDataSource,
    "Hive": HiveDataSource,
    "Oracle": OracleDataSource,
}

def get_config_path():
    return os.path.join(os.path.dirname(__file__), "config", "config.yaml")

def build_config(data_ingestion, llm_inference):
    return {
        "data_ingestion": data_ingestion,
        "llm_inference": llm_inference,
    }

def show_workflow_selector():
    st.sidebar.markdown("### Workflow Management")
    workflows = list_workflows()
    workflow_ids = [w["workflow_id"] for w in workflows]
    selected = st.sidebar.selectbox("Select Workflow", ["<New Workflow>"] + workflow_ids)
    return selected, workflows

def show_config_editor(config, section, options):
    st.sidebar.markdown(f"#### {section.replace('_', ' ').title()} Config")
    params = {}
    for key, val in options.items():
        if isinstance(val, dict):
            params[key] = show_config_editor(config.get(key, {}), key, val)
        else:
            default = config.get(key, val)
            if isinstance(val, int):
                params[key] = st.sidebar.number_input(f"{section}: {key}", value=default)
            elif isinstance(val, float):
                params[key] = st.sidebar.number_input(f"{section}: {key}", value=default, format="%f")
            elif isinstance(val, bool):
                params[key] = st.sidebar.checkbox(f"{section}: {key}", value=default)
            else:
                params[key] = st.sidebar.text_input(f"{section}: {key}", value=default)
    return params

def main():
    # --- Ensure config and llm_inference are loaded before using them ---
    st.title("Modular RAG Framework (MCP-Ready)")
    st.sidebar.header("Configuration")

    # Workflow selection/creation
    selected_wf, workflows = show_workflow_selector()
    config = None
    data_ingestion = {}
    llm_inference = {}
    try:
        if selected_wf != "<New Workflow>":
            config = load_workflow(selected_wf)
            st.success(f"Loaded workflow: {selected_wf}")
        else:
            config = None
    except Exception as e:
        st.warning(f"Error loading workflow: {e}. Using defaults.")
        config = None

    # --- Ensure llm_inference and data_ingestion are always initialized before use ---
    default_data_ingestion = {
        "datasource": {"type": "CSV", "params": {"path": "transaction_data.csv"}},
        "vectordb": {"type": "ChromaDB", "params": {"persist_directory": "./chromadb"}},
        "embedding": {"type": "OpenAI", "params": {"model": "text-embedding-ada-002", "chunk_size": 512, "rank": 1}},
    }
    default_llm_inference = {
        "llm": {"type": "OpenAI", "params": {"model": "gpt-4o-mini", "max_tokens": 512, "temperature": 0.7, "top_p": 1.0}},
        "prompt_vars": {},
    }

    def deep_update(d, u):
        # Recursively update dict d with values from u (u is default)
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d.setdefault(k, v)
        return d

    if config and isinstance(config, dict):
        data_ingestion = config.get("data_ingestion", {})
        llm_inference = config.get("llm_inference", {})
    data_ingestion = deep_update(data_ingestion, default_data_ingestion)
    llm_inference = deep_update(llm_inference, default_llm_inference)

    # Final fallback: ensure both are dicts with required keys
    if not isinstance(data_ingestion, dict):
        data_ingestion = dict(default_data_ingestion)
    if not isinstance(llm_inference, dict):
        llm_inference = dict(default_llm_inference)
    # Ensure required keys exist
    for k, v in default_data_ingestion.items():
        if k not in data_ingestion:
            data_ingestion[k] = v
    for k, v in default_llm_inference.items():
        if k not in llm_inference:
            llm_inference[k] = v

    # --- MCP Parameters UI ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### MCP Parameters")
    mcp_params = llm_inference.get("mcp", {})
    mcp_enabled = st.sidebar.checkbox("Enable MCP", value=mcp_params.get("enabled", True), key="mcp_enabled_checkbox")
    fallback_to_vectordb = st.sidebar.checkbox("Fallback to Vector DB if MCP context is empty", value=mcp_params.get("fallback_to_vectordb", False), key="mcp_fallback_checkbox")
    mcp_tools = mcp_params.get("tools", [])
    # Only show first tool for editing (extend as needed)
    mcp_tool_params = mcp_tools[0]["params"] if mcp_tools and "params" in mcp_tools[0] else {}
    mcp_user = st.sidebar.text_input("MCP Oracle User", value=mcp_tool_params.get("user", "system"), key="mcp_user_input")
    mcp_password = st.sidebar.text_input("MCP Oracle Password", value=mcp_tool_params.get("password", "PULL FROM ENV:ORACLE_PASS"), type="password", key="mcp_password_input")
    mcp_host = st.sidebar.text_input("MCP Oracle Host", value=mcp_tool_params.get("host", "localhost"), key="mcp_host_input")
    mcp_port = st.sidebar.number_input("MCP Oracle Port", min_value=1, value=int(mcp_tool_params.get("port", 1521)), key="mcp_port_input")
    mcp_service_name = st.sidebar.text_input("MCP Oracle Service Name", value=mcp_tool_params.get("service_name", "FREEPDB1"), key="mcp_service_name_input")
    mcp_query = st.sidebar.text_area("MCP Query", value=mcp_tool_params.get("query", "SELECT * FROM FIN_ANA_DATA FETCH FIRST 5 ROWS ONLY"), key="mcp_query_input")
    mcp_fetch_size = st.sidebar.number_input("MCP Fetch Size", min_value=1, value=int(mcp_tool_params.get("fetch_size", 100)), key="mcp_fetch_size_input")

    # Update llm_inference dict with MCP UI values
    llm_inference["mcp"] = {
        "enabled": mcp_enabled,
        "fallback_to_vectordb": fallback_to_vectordb,
        "tools": [
            {
                "name": "oracle_query",
                "description": "Query Oracle DB using Model Context Protocol",
                "datasource": "oracle",
                "params": {
                    "user": mcp_user,
                    "password": mcp_password,
                    "host": mcp_host,
                    "port": mcp_port,
                    "service_name": mcp_service_name,
                    "query": mcp_query,
                    "fetch_size": mcp_fetch_size,
                },
            }
        ],
    }

    # UI for editing config
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Data Ingestion")
    datasource_type = st.sidebar.selectbox(
        "Data Source", list(DATASOURCES.keys()),
        index=list(DATASOURCES.keys()).index(data_ingestion["datasource"]["type"]) if data_ingestion else 0,
        key="data_source_selectbox_main"
    )
    ds_params = {}
    if datasource_type == "CSV":
        ds_params = {"path": st.sidebar.text_input("CSV Path", value=data_ingestion["datasource"]["params"].get("path", "transaction_data.csv"))}
    elif datasource_type == "RDBMS":
        ds_params = {"conn_str": st.sidebar.text_input("RDBMS Connection String", value=data_ingestion["datasource"]["params"].get("conn_str", ""))}
    elif datasource_type == "Hive":
        ds_params = {"conn_str": st.sidebar.text_input("Hive Connection String", value=data_ingestion["datasource"]["params"].get("conn_str", ""))}
    elif datasource_type == "Oracle":
        # Pull Oracle password from env if config uses PULL FROM ENV:VARNAME
        pw_val = data_ingestion["datasource"]["params"].get("password", "")
        if isinstance(pw_val, str) and pw_val.upper().startswith("PULL FROM ENV:"):
            env_var = pw_val.split(":", 1)[1].strip()
            oracle_password = os.environ.get(env_var, "")
        else:
            oracle_password = pw_val
        ds_params = {
            "user": st.sidebar.text_input("Oracle User", value=data_ingestion["datasource"]["params"].get("user", "system")),
            "password": st.sidebar.text_input("Oracle Password", type="password", value=oracle_password),
            "host": st.sidebar.text_input("Oracle Host", value=data_ingestion["datasource"]["params"].get("host", "localhost")),
            "port": st.sidebar.number_input("Oracle Port", min_value=1, value=int(data_ingestion["datasource"]["params"].get("port", 1521))),
            "service_name": st.sidebar.text_input("Oracle Service Name", value=data_ingestion["datasource"]["params"].get("service_name", "FREEPDB1")),
            "table": st.sidebar.text_input("Oracle Table", value=data_ingestion["datasource"]["params"].get("table", "transactions")),
        }
    datasource = DATASOURCES[datasource_type](**ds_params)

    vectordb_type = st.sidebar.selectbox(
        "Vector DB", list(VECTOR_DBS.keys()),
        index=list(VECTOR_DBS.keys()).index(data_ingestion["vectordb"]["type"]),
        key="vectordb_selectbox_main"
    )
    vectordb_params = {"persist_directory": st.sidebar.text_input(
        "ChromaDB Directory",
        value=data_ingestion["vectordb"]["params"].get("persist_directory", "./chromadb"),
        key="chroma_dir_input_main"
    )}
    vectordb = VECTOR_DBS[vectordb_type](**vectordb_params)

    embedding_type = st.sidebar.selectbox(
        "Embedding Type",
        ["openai", "sentence-transformers"],
        index=["openai", "sentence-transformers"].index(data_ingestion["embedding"]["type"].lower()),
        key="embedding_type_selectbox_main"
    )
    embedding_params = {}
    # Dynamically set model default based on embedding_type
    if embedding_type == "sentence-transformers":
        model_default = "all-MiniLM-L6-v2"
    elif embedding_type == "openai":
        model_default = "text-embedding-ada-002"
    else:
        model_default = ""

    # If the embedding type changed, reset the model field to the default
    if (
        "_last_embedding_type" not in st.session_state or
        st.session_state["_last_embedding_type"] != embedding_type
    ):
        st.session_state["embedding_model_value"] = model_default
    else:
        st.session_state.setdefault("embedding_model_value", data_ingestion["embedding"]["params"].get("model", model_default))

    embedding_params["model"] = st.sidebar.text_input(
        "Embedding Model",
        value=st.session_state["embedding_model_value"],
        key=f"embedding_model_input_{embedding_type}"
    )
    st.session_state["_last_embedding_type"] = embedding_type
    embedding_params["chunk_size"] = st.sidebar.number_input(
        "Chunk Size",
        min_value=1,
        value=int(data_ingestion["embedding"]["params"].get("chunk_size", 512)),
        key=f"embedding_chunk_size_{embedding_type}"
    )
    # Add max_batch_size parameter to UI for all embedding types
    embedding_params["max_batch_size"] = st.sidebar.number_input(
        "Max Batch Size",
        min_value=1,
        value=int(data_ingestion["embedding"]["params"].get("max_batch_size", 512)),
        key=f"embedding_max_batch_size_{embedding_type}"
    )
    if embedding_type == "openai":
        embedding_params["rank"] = st.sidebar.number_input(
            "Rank",
            min_value=1,
            value=int(data_ingestion["embedding"]["params"].get("rank", 1)),
            key=f"embedding_rank_{embedding_type}"
        )
        api_key_val = data_ingestion["embedding"]["params"].get("api_key", "")
        if isinstance(api_key_val, str) and api_key_val.upper().startswith("PULL FROM ENV:"):
            env_var = api_key_val.split(":", 1)[1].strip()
            api_key = os.environ.get(env_var, "")
        else:
            api_key = api_key_val
        embedder = OpenAIEmbedding(
            model=embedding_params["model"],
            api_key=api_key,
            chunk_size=embedding_params["chunk_size"],
            rank=embedding_params["rank"],
        )
    else:
        embedder = SentenceTransformersEmbedding(
            model=embedding_params["model"],
            chunk_size=embedding_params["chunk_size"]
        )

    # LLM Inference config
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### LLM Inference")
    llm_type = st.sidebar.selectbox(
        "LLM", list(LLM_MODELS.keys()),
        index=list(LLM_MODELS.keys()).index(llm_inference["llm"]["type"]),
        key="llm_selectbox_main"
    )
    llm_params = {
        "model": st.sidebar.text_input(
            "OpenAI LLM Model",
            value=llm_inference["llm"]["params"].get("model", "gpt-4o-mini"),
            key="openai_llm_model_input_main"
        ),
        "max_tokens": st.sidebar.number_input("Max Tokens", min_value=32, max_value=4096, value=int(llm_inference["llm"]["params"].get("max_tokens", 512)), key="max_tokens_input_main"),
        "temperature": st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=float(llm_inference["llm"]["params"].get("temperature", 0.7)), key="temperature_slider_main"),
        "top_p": st.sidebar.slider(
            "top_p (nucleus sampling): Cumulative probability threshold (e.g., 0.9). Model samples from smallest set of tokens whose total probability ≥ top_p.",
            min_value=0.0, max_value=1.0, value=float(llm_inference["llm"]["params"].get("top_p", 1.0)),
            key="top_p_slider_main"
        ),
    }
    # MCP fallback config
    mcp_fallback_to_vectordb = llm_inference.get("mcp", {}).get("fallback_to_vectordb", True)
    mcp_fallback_to_vectordb = st.sidebar.checkbox("MCP: Fallback to Vector DB if MCP context is empty", value=mcp_fallback_to_vectordb)

    # top_k for RAG pipeline query
    rag_query_top_k = llm_inference.get("query", {}).get("top_k", 10)
    top_k = st.sidebar.number_input(
        "top_k: Only the top k most likely tokens are considered for next-token sampling.",
        min_value=1, value=int(rag_query_top_k)
    )
    # Save in config for workflow
    if "mcp" not in llm_inference:
        llm_inference["mcp"] = {}
    llm_inference["mcp"]["fallback_to_vectordb"] = mcp_fallback_to_vectordb
    api_key_val = llm_inference["llm"]["params"].get("api_key", "")
    if isinstance(api_key_val, str) and api_key_val.upper().startswith("PULL FROM ENV:"):
        env_var = api_key_val.split(":", 1)[1].strip()
        llm_api_key = os.environ.get(env_var, "")
    else:
        llm_api_key = api_key_val
    llm = OpenAILLM(
        model=llm_params["model"],
        api_key=llm_api_key,
        max_tokens=llm_params["max_tokens"],
        temperature=llm_params["temperature"],
        top_p=llm_params["top_p"],
    )

    # Prompt variables
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Prompt Template Variables**")
    if "prompt_var_keys" not in st.session_state:
        st.session_state["prompt_var_keys"] = list(llm_inference.get("prompt_vars", {}).keys())
        st.session_state["prompt_var_values"] = dict(llm_inference.get("prompt_vars", {}))
    # Remove variable
    remove_var = st.sidebar.selectbox("Remove variable", [""] + st.session_state["prompt_var_keys"])
    if remove_var and st.sidebar.button("Remove Selected Variable"):
        if remove_var in st.session_state["prompt_var_keys"]:
            st.session_state["prompt_var_keys"].remove(remove_var)
            st.session_state["prompt_var_values"].pop(remove_var, None)
    # Edit existing variables
    for var in st.session_state["prompt_var_keys"]:
        st.session_state["prompt_var_values"][var] = st.sidebar.text_input(
            f"Prompt variable: {var}",
            value=str(st.session_state["prompt_var_values"].get(var, "")),
        )
    # Add new variable
    new_var = st.sidebar.text_input("Add new prompt variable (name)", value="")
    new_val = st.sidebar.text_input("Value for new variable", value="")
    if (
        new_var
        and new_var not in st.session_state["prompt_var_keys"]
        and st.sidebar.button("Add Variable")
    ):
        st.session_state["prompt_var_keys"].append(new_var)
        st.session_state["prompt_var_values"][new_var] = new_val
    prompt_vars = dict(st.session_state["prompt_var_values"])

    # Workflow ID for saving
    st.sidebar.markdown("---")
    workflow_id = st.sidebar.text_input("Workflow ID", value=selected_wf if selected_wf != "<New Workflow>" else "")
    if st.sidebar.button("Save Workflow"):
        config_to_save = build_config(
            {
                "datasource": {"type": datasource_type, "params": ds_params},
                "vectordb": {"type": vectordb_type, "params": vectordb_params},
                "embedding": {"type": embedding_type, "params": embedding_params},
            },
            {
                "llm": {"type": llm_type, "params": llm_params},
                "prompt_vars": prompt_vars,
            },
        )
        if workflow_id:
            # Save workflow with user-specified workflow_id if provided, else generate new
            from config.workflow_utils import WORKFLOWS_DIR
            import yaml, os
            os.makedirs(WORKFLOWS_DIR, exist_ok=True)
            workflow_path = os.path.join(WORKFLOWS_DIR, f"{workflow_id}.yaml")
            with open(workflow_path, "w") as f:
                yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)
            st.success(f"Workflow '{workflow_id}' saved!")
        else:
            st.warning("Please provide a workflow ID to save.")

    # Build pipeline for current config
    pipeline = RAGPipeline(
        datasource,
        embedder,
        vectordb,
        llm,
        mcp_fallback_to_vectordb=mcp_fallback_to_vectordb
    )
    st.session_state["pipeline"] = pipeline

    # --- Main UI: Run workflows ---
    st.header("Run Workflow")
    st.markdown("Select which part of the workflow to run:")
    run_ingest = st.checkbox("Run Data Ingestion (Load to Vector DB)", value=True)
    run_infer = st.checkbox("Run LLM Inference", value=True)
    st.markdown("---")

    if st.button("Run Selected Workflow Steps"):
        if not pipeline:
            st.warning("Please configure and save a workflow first!")
        else:
            if run_ingest:
                with st.spinner("Running data ingestion (building vector store)..."):
                    pipeline.build_vector_store()
                st.success("Data ingestion complete!")
            if run_infer:
                st.subheader("Ask a Question")
                user_query = st.text_input("Your question:", key="user_query")
                if user_query:
                    with st.spinner("Generating answer..."):
                        answer = pipeline.query(user_query, prompt_vars=prompt_vars)
                    st.markdown(f"**Answer:** {answer}")
                else:
                    st.info("Enter a question above to run inference.")

    # Show all workflows
    st.markdown("---")
    st.markdown("### Existing Workflows")

    for wf in workflows:
        st.markdown(f"- **{wf['workflow_id']}**")


if __name__ == "__main__":
    main()



