# Ontario Tech University FAQ Chatbot using LangChain and OpenAI

LLM-powered Q&A chatbot built on Ontario Tech University's admissions and registrar FAQs.
<img width="973" height="585" alt="image" src="https://github.com/user-attachments/assets/c3304201-ff4a-4752-9961-25c02b5a258f" />
<img width="992" height="516" alt="image" src="https://github.com/user-attachments/assets/6ca181ec-f407-497e-a83a-4143f409fd29" />



# Overview
This project leverages Large Language Models (LLMs) using LangChain and OpenAI's GPT-4o-mini to create a question-and-answer system based on FAQs scraped from Ontario Tech University's admissions and registrar websites. It utilizes HuggingFace embeddings for text vectorization and FAISS for efficient vector database storage. The system is designed to provide accurate answers to university-related queries while gracefully handling out-of-scope questions without hallucinating incorrect information. A user-friendly interface is provided by Streamlit.

# Key Features
* **Real-world Data:** FAQs scraped directly from Ontario Tech University's admissions and registrar websites across 12 sections including admissions, fees, financial aid, OSAP, registration, and convocation.
* **Custom Web Scraper:** Includes a Python scraper (`scrape_ontario_tech_faqs.py`) to pull and update FAQ data from the university's website.
* **LangChain + OpenAI GPT-4o-mini:** Employs LLMs for a robust and cost-efficient Q&A system.
* **Relevance Filtering:** Uses FAISS similarity scores to block off-topic questions before they reach the LLM, saving API tokens.
* **Streamlit Interface:** Provides an intuitive UI for users to submit questions and receive answers.
* **HuggingFace Embeddings:** Uses `sentence-transformers/all-MiniLM-L6-v2` for accurate vector representation of text.
* **FAISS Vector Database:** Stores text embeddings in a locally hosted vector database for efficient answer retrieval.

# Installation
* Clone this repo
* `cd` into the working directory
* Create and activate a virtual environment:
  ```bash
  python -m venv rag-env
  rag-env\Scripts\activate  # Windows
  source rag-env/bin/activate  # Mac/Linux
  ```
* Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
* Create a `.env` file in the root directory and add your OpenAI API key:
  ```
  OPENAI_API_KEY=your_key_here
  ```

# Usage
1. (Optional) Re-scrape the latest FAQs from Ontario Tech's website:
   ```bash
   python scrape_ontario_tech_faqs.py
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```
3. Type your question into the input field. The system will search the vector database for the most relevant FAQ answer. If the question is outside the scope of Ontario Tech University topics, the chatbot will respond accordingly without calling the LLM.

# Project Structure
```
├── langchain_helper.py          # LangChain logic, FAISS vector DB, OpenAI integration
├── main.py                      # Streamlit application
├── scrape_ontario_tech_faqs.py  # Web scraper to pull FAQs from Ontario Tech's website
├── ontario_tech_faqs_all.csv    # Scraped FAQ data (Section, Question, Answer)
├── requirements.txt             # Project dependencies
├── .env                         # API keys (not committed to GitHub)
└── .gitignore                   # Excludes .env, faiss_index/, rag-env/ from version control
```
