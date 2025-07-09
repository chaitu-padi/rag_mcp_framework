
from mcp_tools import mcp_registry


class RAGPipeline:
    """
    Modular RAG Pipeline for loading data, generating embeddings, storing in vector DB, and querying LLMs.
    Supports Model Context Protocol (MCP) for context retrieval from Oracle or other sources.
    """
    def __init__(
        self,
        datasource,
        embedder,
        vectordb,
        llm,
        prompt_template=None,
        mcp_tools_enabled=False,
        mcp_tool_name=None,
        mcp_tool_params=None,
        mcp_fallback_to_vectordb=True,
    ):
        """
        Initialize the RAGPipeline.

        Args:
            datasource: Data source object with a .load() method.
            embedder: Embedding model object with an .embed() method.
            vectordb: Vector DB object with .add() and .query() methods.
            llm: LLM object with a .generate() method.
            prompt_template: Optional prompt template string.
            mcp_tools_enabled: Enable MCP context retrieval.
            mcp_tool_name: Name of the MCP tool to use.
            mcp_tool_params: Parameters for the MCP tool.
            mcp_fallback_to_vectordb: If True, fallback to vector DB if MCP context is empty. If False, do not fallback.
        """
        self.datasource = datasource
        self.embedder = embedder
        self.vectordb = vectordb
        self.llm = llm
        self.prompt_template = (
            prompt_template or "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        self.mcp_tools_enabled = mcp_tools_enabled
        self.mcp_tool_name = mcp_tool_name
        self.mcp_tool_params = mcp_tool_params or {}
        self.mcp_fallback_to_vectordb = mcp_fallback_to_vectordb

    def build_vector_store(self):
        """
        Loads documents, generates embeddings in batches, and adds them to the vector DB.
        Embedding and vector DB add operations are batched to avoid exceeding max_batch_size.
        Logs progress and timing information.
        """
        import time
        import multiprocessing

        print("[RAGPipeline] Loading documents...")
        docs = self.datasource.load()
        print(f"[RAGPipeline] Loaded {len(docs)} documents: {docs[:2]} ...")
        print("[RAGPipeline] Generating embeddings in batches...")

        start_time = time.time()

        # Determine max_batch_size from embedder config or attribute, fallback to default
        max_batch_size = 5461
        if hasattr(self.embedder, "config") and isinstance(self.embedder.config, dict):
            max_batch_size = int(self.embedder.config.get("max_batch_size", max_batch_size))
        elif hasattr(self.embedder, "max_batch_size"):
            max_batch_size = int(getattr(self.embedder, "max_batch_size", max_batch_size))

        # Dynamically choose batch size based on number of docs and CPU count
        cpu_count = multiprocessing.cpu_count()
        batch_size = min(max(32, len(docs) // (cpu_count * 4)), max_batch_size, len(docs))

        # Generate embeddings in batches with progress bar and timing
        from tqdm import tqdm
        embeddings = []
        num_batches = (len(docs) + max_batch_size - 1) // max_batch_size
        print(f"[RAGPipeline] Generating embeddings in {num_batches} batches of up to {max_batch_size}...")
        batch_times = []
        for i in tqdm(range(0, len(docs), max_batch_size), desc="Embedding batches", unit="batch"):
            batch_start = time.time()
            batch = docs[i : i + max_batch_size]
            embeddings.extend(self.embedder.embed(batch))
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            print(f"[RAGPipeline] Batch {i//max_batch_size+1}/{num_batches} done in {batch_time:.2f}s, total embeddings: {len(embeddings)}")
        if batch_times:
            print(f"[RAGPipeline] Average batch time: {sum(batch_times)/len(batch_times):.2f}s")

        total_time = time.time() - start_time
        print(f"[RAGPipeline] Generated {len(embeddings)} embeddings in {total_time:.2f} seconds. Example: {embeddings[0] if embeddings else 'None'}")

        print("[RAGPipeline] Adding embeddings to vector DB in batches...")
        for i in range(0, len(embeddings), max_batch_size):
            emb_batch = embeddings[i : i + max_batch_size]
            doc_batch = docs[i : i + max_batch_size]
            self.vectordb.add(emb_batch, doc_batch)
        print("[RAGPipeline] Vector store build complete.")

    def query(self, user_query, top_k=None, **extra_vars):
        """
        Answers a user query using the RAG pipeline.
        If MCP is enabled and configured, uses MCP for context retrieval from Oracle.
        Otherwise, retrieves context from the vector DB using embeddings.

        Args:
            user_query: The user's question.
            top_k: Number of top documents to retrieve from the vector DB.
            **extra_vars: Additional variables for prompt formatting.

        Returns:
            LLM-generated answer string.
        """
        context = None
        mcp_result = None

        # Try MCP context retrieval if enabled
        if self.mcp_tools_enabled and self.mcp_tool_name:
            # Prepare params for MCP tool, mapping user_query to 'query' if needed
            mcp_params = dict(self.mcp_tool_params) if self.mcp_tool_params else {}
            # If the tool is 'oracle_query', ensure all required connection params are present
            if self.mcp_tool_name == "oracle_query":
                # Try to get connection params from data_ingestion.datasource.params if not present
                required_keys = ["user", "password", "host", "port", "service_name"]
                import os
                for key in required_keys:
                    if key not in mcp_params and hasattr(self, "datasource") and self.datasource:
                        # Try to get from datasource config if available
                        val = getattr(self.datasource, key, None)
                        if val is None and hasattr(self.datasource, "config"):
                            val = self.datasource.config.get(key)
                        if val is None and hasattr(self.datasource, "params"):
                            val = self.datasource.params.get(key)
                        if val is not None:
                            mcp_params[key] = val
                    # If the key is 'password' and the value is a PULL FROM ENV pattern, resolve from env
                    if key == "password" and key in mcp_params:
                        pw_val = mcp_params[key]
                        if isinstance(pw_val, str) and pw_val.upper().startswith("PULL FROM ENV:"):
                            env_var = pw_val.split(":", 1)[1].strip()
                            mcp_params[key] = os.environ.get(env_var)
                # If still missing, try to get from self.embedder.config (for legacy)
                for key in required_keys:
                    if key not in mcp_params and hasattr(self.embedder, "config"):
                        val = self.embedder.config.get(key)
                        if val is not None:
                            mcp_params[key] = val
                # If 'query' is not set, use user_query as SQL
                if "query" not in mcp_params:
                    mcp_params["query"] = user_query
            print(f"[RAGPipeline] MCP enabled. Calling tool: {self.mcp_tool_name} ")#with params: {mcp_params}
            try:
                mcp_result = mcp_registry.call(
                    self.mcp_tool_name, **mcp_params
                )
                print(f"[RAGPipeline] MCP tool raw result: {repr(mcp_result)} (type: {type(mcp_result)})")
                context = str(mcp_result)
                print(f"[RAGPipeline] MCP context (as string): {repr(context)}")
            except Exception as e:
                print(f"[RAGPipeline] MCP tool error: {e}")


        # Fallback to vector DB retrieval if MCP is not used or fails, and fallback is enabled
        if not context:
            if self.mcp_tools_enabled and not self.mcp_fallback_to_vectordb:
                print("[RAGPipeline] MCP context empty or failed, and fallback to vector DB is disabled. Returning empty or error context.")
                context = "[No context retrieved from MCP]"
            else:
                print("[RAGPipeline] MCP context empty or failed, falling back to vector DB retrieval.")
                query_emb = self.embedder.embed([user_query])[0]
                # Always use a safe top_k (default 10) if not provided
                safe_top_k = top_k if top_k is not None else 10
                retrieved_docs = self.vectordb.query(query_emb, top_k=safe_top_k)
                print(f"[RAGPipeline] Retrieved {len(retrieved_docs)} docs from vector DB.")
                context = "\n".join(retrieved_docs)

        # Format prompt with context and user question
        prompt_vars = dict(context=context, question=user_query)
        prompt_vars.update(extra_vars)
        prompt = self.prompt_template.format(**prompt_vars)
        print(f"[RAGPipeline] Final prompt sent to LLM: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        return self.llm.generate(prompt)
