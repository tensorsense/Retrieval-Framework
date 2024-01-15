from pathlib import Path
from typing import List

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import TextLoader

import config


root_path = Path(__file__).parent.resolve()
library_path = root_path / "library" / "fulltext"
db_path = root_path / "vectorstore"


def load_library() -> List[Document]:
    docs = []
    for txt_path in library_path.glob("*.txt"):
        loader = TextLoader(txt_path.as_posix())
        docs.extend(loader.load())
    return docs


def build_simple_retriever() -> BaseRetriever:
    if db_path.exists():
        print(f"Loading vectorstore from disk: {db_path.as_posix()}")
        vectorstore = Chroma(persist_directory=db_path.as_posix(), embedding_function=config.openai_emb_model)
    else:
        print(f"Creating a vectorstore and writing it to: {db_path.as_posix()}")
        docs = load_library()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=config.openai_emb_model, persist_directory=db_path.as_posix())
    retriever = vectorstore.as_retriever()
    return retriever


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    retriever = build_simple_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | config.openai_chat_model
            | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    chain = build_rag_chain()
    print(chain.invoke("What does AU4 represent?"))
