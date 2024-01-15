from pathlib import Path
from typing import List

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import TextLoader

from demo import config

# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv("azure.env"))

root_path = Path(__file__).parent.resolve()
library_path = root_path / "library"
db_path = root_path / "vectorstore"

embedding = AzureOpenAIEmbeddings(
    azure_deployment=config.AZURE_OPENAI_DEPLOYMENT_EMBEDDING,
    openai_api_version="2023-07-01-preview",
)

llm = AzureChatOpenAI(
    temperature=0,
    azure_deployment=config.AZURE_OPENAI_DEPLOYMENT_BASE_RETREIVER,
    openai_api_version="2023-07-01-preview",
)


def load_library() -> List[Document]:
    docs = []
    for txt_path in library_path.glob("*.txt"):
        loader = TextLoader(txt_path.as_posix())
        docs.extend(loader.load())
    return docs


def build_simple_retriever() -> BaseRetriever:
    if db_path.exists():
        print(f"Loading vectorstore from disk: {db_path.as_posix()}")
        vectorstore = Chroma(persist_directory=db_path.as_posix(), embedding_function=embedding)
    else:
        print(f"Creating a vectorstore and writing it to: {db_path.as_posix()}")
        docs = load_library()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory=db_path.as_posix())
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
            | llm
            | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    chain = build_rag_chain()
    print(chain.invoke("What does AU4 represent?"))
