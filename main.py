import streamlit as st 
from langchain_helper import create_vector_db, get_question_answer_chain

st.title("Durham College Frequently Asked Questions and Answers")

Question = st.text_input("Question: ")

if Question:
    chain = get_question_answer_chain()
    response = chain(Question)
    
    st.header("Answer: ")
    st.write(response["result"])
