import chromadb
from vectordb.base import VectorDB


class ChromaDB(VectorDB):
    def get_all_documents(self):
        # Retrieve all documents from the collection
        # ChromaDB collections support .get() with no arguments to get all
        results = self.collection.get()
        return results.get("documents", [])

    def __init__(self, persist_directory="./chromadb"):
        # Use PersistentClient for persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("rag_docs")

    def add(self, embeddings, documents):
        # Chroma expects lists for ids, embeddings, documents
        ids = [str(i) for i in range(len(documents))]
        # Filter out any bad embeddings/documents
        clean_ids, clean_embeddings, clean_documents = [], [], []
        for i, (emb, doc) in enumerate(zip(embeddings, documents)):
            if (
                emb is not None
                and isinstance(emb, list)
                and all(isinstance(x, (float, int)) for x in emb)
                and doc
                and isinstance(doc, str)
            ):
                clean_ids.append(ids[i])
                clean_embeddings.append([float(x) for x in emb])
                clean_documents.append(doc)
        if not clean_ids:
            raise ValueError("No valid embeddings/documents to add to ChromaDB.")
        print(f"[ChromaDB.add] Cleaned: {len(clean_ids)} records")
        try:
            self.collection.add(
                ids=clean_ids, embeddings=clean_embeddings, documents=clean_documents
            )
            print("self.collection.count():", self.collection.count())
        except Exception as e:
            if "capture() takes 1 positional argument but 3 were given" in str(e):
                print(
                    "[ChromaDB.add] Telemetry bug encountered, but data likely added. Continuing..."
                )
            else:
                print("[ChromaDB.add] Failed to add to collection:", e)
                import traceback

                traceback.print_exc()
                print(
                    f"ids type: {type(clean_ids)}, embeddings type: {type(clean_embeddings)}, documents type: {type(clean_documents)}"
                )
                print(f"ids sample: {clean_ids[:2]}")
                print(f"embeddings sample: {clean_embeddings[:2]}")
                print(f"documents sample: {clean_documents[:2]}")
                raise

    def query(self, embedding, top_k=5):
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        return results["documents"][0]
