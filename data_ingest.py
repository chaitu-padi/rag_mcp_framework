import os
import yaml
from datasources.csv_source import CSVDataSource
from datasources.hive_source import HiveDataSource
from datasources.rdbms_source import RDBMSDataSource
from datasources.oracle_source import OracleDataSource
from embeddings.openai_embed import OpenAIEmbedding
from vectordb.chromadb import ChromaDB

# Load config
def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

def get_datasource(cfg):
    ds_type = cfg["datasource"]["type"]
    ds_params = cfg["datasource"]["params"]
    if ds_type == "csv":
        return CSVDataSource(**ds_params)
    elif ds_type == "rdbms":
        return RDBMSDataSource(**ds_params)
    elif ds_type == "hive":
        return HiveDataSource(**ds_params)
    elif ds_type == "oracle":
        return OracleDataSource(**ds_params)
    else:
        raise ValueError(f"Unsupported datasource type: {ds_type}")

def get_embedder(cfg):
    embedding_cfg = cfg["embedding"]
    embedding_params = embedding_cfg["params"].copy()
    api_key_val = embedding_params.get("api_key", "")
    if isinstance(api_key_val, str) and api_key_val.upper().startswith("PULL FROM ENV:"):
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

def get_vectordb(cfg):
    vectordb_cfg = cfg["vectordb"]
    if vectordb_cfg["type"] == "chromadb":
        return ChromaDB(**vectordb_cfg["params"])
    else:
        raise ValueError(f"Unsupported vectordb type: {vectordb_cfg['type']}")

def main():
    config = load_config()
    data_ingest_cfg = config["data_ingestion"]
    datasource = get_datasource(data_ingest_cfg)
    embedder = get_embedder(data_ingest_cfg)
    vectordb = get_vectordb(data_ingest_cfg)
    from rag.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline(datasource, embedder, vectordb, llm=None)
    pipeline.build_vector_store()
    print("[Data Ingestion] Vector store build complete.")

if __name__ == "__main__":
    main()
