import sys
import os
import tempfile
import shutil
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

import gradio as gr
from pypdf import PdfReader
from PIL import Image
import pytesseract
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import GROQ_API_KEY

import platform
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4

# loaded once at startup, shared across all sessions since it's read-only
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

    elif ext in [".jpg", ".jpeg", ".png"]:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)

    else:
        return ""

    return text.strip()


def build_vectorstore_from_texts(all_texts: list, tmp_dir: str):
    # combining all uploaded files into one set of chunks
    docs = [Document(page_content=t) for t in all_texts if t.strip()]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=tmp_dir
    )
    return vectorstore


def build_chain(retriever):
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

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def cleanup_session(session: dict):
    # close chroma connection before deleting files, Windows holds locks otherwise
    if session.get("vectorstore"):
        try:
            session["vectorstore"]._client.close()
        except Exception:
            pass

    if session.get("tmp_dir") and os.path.exists(session["tmp_dir"]):
        try:
            shutil.rmtree(session["tmp_dir"])
        except PermissionError:
            # Windows sometimes still holds the lock briefly, safe to ignore
            pass


def process_files(files, session: dict):
    if not files:
        return "No files uploaded.", session

    # clean up this user's previous session before starting fresh
    cleanup_session(session)

    all_texts = []
    file_names = []

    for file in files:
        file_path = str(file)
        text = extract_text(file_path)
        if text:
            all_texts.append(text)
            file_names.append(os.path.basename(file_path))

    if not all_texts:
        return "Could not extract text from any of the uploaded files.", session

    # each session gets its own unique temp directory so users don't overlap
    tmp_dir = tempfile.mkdtemp(prefix=f"rag_{uuid.uuid4().hex}_")
    vectorstore = build_vectorstore_from_texts(all_texts, tmp_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    total_chars = sum(len(t) for t in all_texts)

    # storing everything in the session dict, not a global variable
    session["chain"] = build_chain(retriever)
    session["retriever"] = retriever
    session["tmp_dir"] = tmp_dir
    session["vectorstore"] = vectorstore

    status = f"Processed {len(file_names)} file(s): {', '.join(file_names)}\n{total_chars} total characters extracted. You can start asking questions."
    return status, session


def chat(message, history, session: dict):
    if not session.get("chain"):
        history.append({"role": "assistant", "content": "Please upload at least one file before asking questions."})
        return "", history, session

    if not message.strip():
        return "", history, session

    answer = session["chain"].invoke(message)
    print(f"session keys: {session.keys()}")
    print(f"retriever: {session.get('retriever')}")
    source_docs = session["retriever"].invoke(message)
    print(f"source docs count: {len(source_docs)}")
    print(f"first doc metadata: {source_docs[0].metadata if source_docs else 'none'}")

    # pulling source chunks separately to show which file the answer came from
    source_docs = session["retriever"].invoke(message)
    sources = list(set(
        os.path.basename(doc.metadata.get("source", "")).replace("_", " ")
        for doc in source_docs
        if doc.metadata.get("source")
    ))

    # only show sources if we actually found some
    if sources:
        source_text = "\n\n**Sources:** " + ", ".join(sources)
        answer += source_text

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return "", history, session


def clear_chat(session: dict):
    cleanup_session(session)
    # reset this user's session completely
    fresh_session = {"chain": None, "retriever": None, "tmp_dir": None, "vectorstore": None}
    return [], "", fresh_session


with gr.Blocks(title="RAG Chatbot") as demo:
    gr.Markdown("## RAG Chatbot")
    gr.Markdown("Upload one or more PDFs, JPGs, or PNGs and ask questions across all of them.")

    # each user gets their own session state, Gradio handles this per browser tab
    session = gr.State({"chain": None, "retriever": None, "tmp_dir": None, "vectorstore": None})

    with gr.Row():
        file_input = gr.File(
            label="Upload your files",
            file_types=[".pdf", ".jpg", ".jpeg", ".png"],
            file_count="multiple"
        )
        upload_status = gr.Textbox(
            label="Status",
            interactive=False,
            lines=3
        )

    file_input.change(process_files, inputs=[file_input, session], outputs=[upload_status, session])

    chatbot = gr.Chatbot(height=450)
    msg = gr.Textbox(
        placeholder="Ask a question about your documents...",
        show_label=False
    )

    with gr.Row():
        submit = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear Chat")

    msg.submit(chat, [msg, chatbot, session], [msg, chatbot, session])
    submit.click(chat, [msg, chatbot, session], [msg, chatbot, session])
    clear.click(clear_chat, inputs=[session], outputs=[chatbot, msg, session])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())