import streamlit as st
from langchain_helper import ensure_vector_db, answer_question

st.title("Durham College FAQ Chatbot: Ask me anything!")


@st.cache_resource(show_spinner="Building vector database (first run only)...")
def load_vectordb():
    return ensure_vector_db()


vectordb = load_vectordb()

question = st.text_input("Question:")

if question:
    with st.spinner("Thinking..."):
        answer = answer_question(question, vectordb)

    st.header("Answer:")
    st.write(answer)
