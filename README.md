# LLM-end-to-end using Langchain and Open Ai
LLM Q&A Project with Durham College FAQs
![DurhamCollegeFAQ2](https://github.com/mohmeez/LLM-end-to-end/assets/96016404/b6b95c2f-f349-4343-912d-264ee6e4709e)
![DurhamCollegeFAQ1](https://github.com/mohmeez/LLM-end-to-end/assets/96016404/5b08212e-939f-43c9-a245-3b1790eb8d28)

# Overview
This project leverages the power of Large Language Models (LLMs) using Langchain and Open Ai to create a question-and-answer system based on sample FAQs from Durham College's website. It utilizes Huggingface's instructor embeddings for text vectorization and FAISS for efficient vector database storage. The system is designed to provide accurate answers to user queries without hallucinating incorrect information. A user-friendly interface is provided by Streamlit, allowing users to interact with the model seamlessly.

# Key Features
* Real-world Data: Utilizes a CSV file containing sample FAQs from Durham College's website.
* Langchain + Open Ai: Employs LLMs for a robust Q&A system.
* Streamlit Interface: Provides an intuitive UI for users to submit questions and receive answers.
* Huggingface Instructor Embeddings: Uses text embeddings for accurate vector representation of text.
* FAISS Vector Database: Stores text embeddings in a locally hosted vector database for efficient retrieval of answers.

# Installation
* clone this repo
* cd into the working directory
* pip install the requirements.txt file
* Get a google api key from google and put it in the .env file

# Usage
To use this project, simply run the Streamlit app by typing (streamlit run main.py) in the terminal. This opens the web application where you can input your question. The system will search the vector database for the closest matching answer from the Durham College FAQs. If a relevant answer is found, it will be displayed; otherwise, the system will respond with "I do not know."

# Project Structure
* langchain_helper.py --> This has the barebones of lanchain code.
* main.py --> This has all the streamlit application script.
* requirements.txt --> This has a list of all the dependencies required.
  
