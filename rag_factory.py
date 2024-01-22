import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.base_query_engine import BaseQueryEngine


def build_query_engine(
        library_dir: str,
        persist_dir: str = "./storage"
) -> BaseQueryEngine:
    # check if storage already exists
    if not os.path.exists(persist_dir):
        # load the documents and create the index
        documents = SimpleDirectoryReader(library_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        # store it for later
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)

    # either way we can now query the index
    query_engine = index.as_query_engine()

    return query_engine
