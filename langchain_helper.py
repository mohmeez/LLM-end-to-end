import os
from dotenv import load_dotenv

from openai import OpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
load_dotenv()


# 1) Try environment variable (for local use)
# 2) Fall back to Streamlit secrets (for Streamlit Cloud)
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not api_key:
    # Helpful error if you forgot to set it anywhere
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Set it in a .env file (local) or in Streamlit Secrets (cloud)."
    )

client = OpenAI(api_key=api_key)

# Where the FAISS index will be stored
VECTOR_DB_FILE_PATH = "faiss_index"

# Smaller, faster embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embeddings():
    """Create the embedding model (same config for save/load)."""
    return HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def create_vector_db():
    """Create FAISS index from the CSV and save it locally."""
    loader = CSVLoader(
        file_path="Durham_College_FAQ.csv",
        encoding="utf-8",
    )
    docs = loader.load()

    embeddings = _get_embeddings()
    vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
    vectordb.save_local(VECTOR_DB_FILE_PATH)

    return vectordb


def ensure_vector_db():
    """
    Load existing FAISS index if present; otherwise build a new one.
    Streamlit will call this once and reuse the result.
    """
    embeddings = _get_embeddings()

    if os.path.isdir(VECTOR_DB_FILE_PATH):
        return FAISS.load_local(
            VECTOR_DB_FILE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        return create_vector_db()


def answer_question(question: str, vectordb) -> str:
    """Retrieve relevant docs and ask OpenAI for an answer."""
    # Directly use FAISS similarity search
    docs = vectordb.similarity_search(question, k=4)

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
    You are an assistant answering questions about Durham College.

    Use the context below when it is relevant. If the context does not give
    an exact answer, reply with a short, generic but helpful answer based on
    typical college practices. Do NOT say "I don't know" and do NOT invent
    specific names, numbers, or policies.

    Context:
    {context}

    User question: {question}

    Answer clearly in 2–4 sentences:
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=300,
    )

    return response.output_text.strip()



