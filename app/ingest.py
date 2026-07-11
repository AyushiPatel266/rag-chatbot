from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHUNK_SIZE, CHUNK_OVERLAP
import os

DOCS_PATH = "./docs"
CHROMA_PATH = "./chroma_db"


def load_documents():
    # loads every PDF in /docs, DirectoryLoader handles multiple files at once
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"loaded {len(documents)} pages from your docs")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    # free local embeddings, runs on your machine, no API key needed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # saves everything to chroma_db/ on disk
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("vectorstore saved to disk, ingestion done")
    return vectorstore


if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    build_vectorstore(chunks)