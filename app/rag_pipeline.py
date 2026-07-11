import sys
import os

sys.path.append(os.path.dirname(__file__))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import GROQ_API_KEY, TOP_K

CHROMA_PATH = "./chroma_db"


def load_vectorstore():
    # same model as ingest.py, has to match or retrieval breaks
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore


def build_rag_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    prompt = PromptTemplate.from_template("""
    Use the context below to answer the question.
    If the answer isn't in the context, just say you don't know, don't guess.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )

    # LCEL chain, this is the modern LangChain way of building pipelines
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


rag_chain, retriever = build_rag_chain()


def get_answer(question: str) -> dict:
    answer = rag_chain.invoke(question)

    # pulling source docs separately so we can show citations in the UI
    source_docs = retriever.invoke(question)
    sources = list(set(
        doc.metadata.get("source", "unknown") for doc in source_docs
    ))

    return {
        "answer": answer,
        "sources": sources
    }