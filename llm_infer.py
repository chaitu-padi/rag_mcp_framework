import os
import yaml
from llms.openai_llm import OpenAILLM
from rag.rag_pipeline import RAGPipeline
from vectordb.chromadb import ChromaDB
from datasources.oracle_source import OracleDataSource
from embeddings.openai_embed import OpenAIEmbedding

def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

def get_vectordb(cfg):
    vectordb_cfg = cfg["vectordb"]
    if vectordb_cfg["type"] == "chromadb":
        return ChromaDB(**vectordb_cfg["params"])
    else:
        raise ValueError(f"Unsupported vectordb type: {vectordb_cfg['type']}")
def resolve_env_vars(params):
    # Replace any value like 'PULL FROM ENV:VARNAME' with os.environ["VARNAME"]
    resolved = dict(params)
    for k, v in params.items():
        if isinstance(v, str) and v.startswith("PULL FROM ENV:"):
            env_var = v.split(":", 1)[1]
            resolved[k] = os.environ.get(env_var, "")
    return resolved

def get_embedder(cfg):
    embedding_cfg = cfg["embedding"]
    embedding_params = embedding_cfg["params"].copy()
    # Resolve env vars for OpenAI API key in embedding params
    api_key_val = embedding_params.get("api_key", "")
    if api_key_val.upper().startswith("PULL FROM ENV:"):
        env_var = api_key_val.split(":", 1)[1].strip()
        embedding_params["api_key"] = os.environ.get(env_var)
    embed_model = embedding_params.get("model")
    chunk_size = embedding_params.get("chunk_size", 512)
    rank = embedding_params.get("rank", 1)
    if embedding_cfg["type"] == "openai":
        return OpenAIEmbedding(
            model=embed_model,
            api_key=embedding_params["api_key"],
            chunk_size=chunk_size,
            rank=rank,
        )
    elif embedding_cfg["type"] == "sentence-transformers":
        from embeddings.sentence_transformers_embed import SentenceTransformersEmbedding
        return SentenceTransformersEmbedding(
            model=embed_model,
            chunk_size=chunk_size,
        )
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_cfg['type']}")

def get_llm(cfg):
    llm_params = cfg["llm"]["params"].copy()
    # Resolve env vars for OpenAI API key in LLM params
    api_key_val = llm_params.get("api_key", "")
    if api_key_val.upper().startswith("PULL FROM ENV:"):
        env_var = api_key_val.split(":", 1)[1].strip()
        llm_params["api_key"] = os.environ.get(env_var)
    max_tokens = llm_params.get("max_tokens", 512)
    temperature = llm_params.get("temperature", 0.7)
    top_p = llm_params.get("top_p", 1.0)
    if cfg["llm"]["type"] == "openai":
        return OpenAILLM(
            model=llm_params["model"],
            api_key=llm_params["api_key"],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        raise ValueError(f"Unsupported LLM type: {cfg['llm']['type']}")

def main():
    config = load_config()
    data_ingest_cfg = config["data_ingestion"]
    llm_inf_cfg = config["llm_inference"]
    # Resolve env vars for Oracle password in MCP tool params
    mcp_cfg = llm_inf_cfg.get("mcp", {})
    if mcp_cfg.get("tools"):
        for tool in mcp_cfg["tools"]:
            if "params" in tool:
                tool["params"] = resolve_env_vars(tool["params"])
    # Also resolve in data_ingestion datasource params if needed
    ds_params = data_ingest_cfg["datasource"]["params"]
    data_ingest_cfg["datasource"]["params"] = resolve_env_vars(ds_params)

    vectordb = get_vectordb(data_ingest_cfg)
    embedder = get_embedder(data_ingest_cfg)
    llm = get_llm(llm_inf_cfg)
    # For MCP, you may want to pass mcp config as well
    prompt_config = llm_inf_cfg.get("prompts", {})
    prompt_template = prompt_config.get(
        "default", "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    extra_vars = {}
    if isinstance(prompt_config, dict):
        for key, value in prompt_config.items():
            if key.startswith("var_"):
                extra_vars[key[4:]] = value
    mcp_cfg = llm_inf_cfg.get("mcp", {})
    pipeline = RAGPipeline(
        datasource=None,  # Not needed for inference
        embedder=embedder,
        vectordb=vectordb,
        llm=llm,
        prompt_template=prompt_template,
        mcp_tools_enabled=mcp_cfg.get("enabled", False),
        mcp_tool_name=mcp_cfg.get("tools", [{}])[0].get("name"),
        mcp_tool_params=mcp_cfg.get("tools", [{}])[0].get("params", {}),
        mcp_fallback_to_vectordb=mcp_cfg.get("fallback_to_vectordb", True),
    )
    query_config = llm_inf_cfg.get("query", {})
    default_top_k = query_config.get("top_k", 10)
    print(**extra_vars)
    print("LLM inference ready. Type your question (or 'exit'):")
    while True:
        user_query = input("Ask a question (or 'exit'): ")
        if user_query.lower() == "exit":
            break
        answer = pipeline.query(user_query, top_k=default_top_k, **extra_vars)
        print("Answer:", answer)

if __name__ == "__main__":
    main()
