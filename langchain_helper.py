import os
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA


from dotenv import load_dotenv
load_dotenv()  # load environment variables from .env

# Create OpenAI LLM model
llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.1,
    # change this to another instruct model if needed
    model_name="gpt-5-nano",
)

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large"
)


VECTORDb_FILE_PATH = "faiss_index"

def ensure_vector_db():
    """Create the FAISS index if it doesn't exist yet."""
    if not os.path.exists(VECTORDb_FILE_PATH):
        create_vector_db()

def create_vector_db():
    # Load data from FAQ 
    loader = CSVLoader(file_path='Durham_College_FAQ.csv', source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(VECTORDb_FILE_PATH)


def get_question_answer_chain():
    ensure_vector_db()
    vectordb = FAISS.load_local(VECTORDb_FILE_PATH, instructor_embeddings, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)
    
    prompt_template = """Given the folowing context and question, please provide an answer based on the context and do not make up any random answers.
    provide as much text as possible from "response" section in the source document context without making as many changes.
    If the answer is not found in the context, kindly state: "I don't know". Dont make up any answers.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain_type_kwargs = {"prompt": PROMPT}
    
    
    chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    input_key="query",
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT})
    
    return chain 


# if __name__ == "__main__":
#     chain = get_question_answer_chain()
    
#     print(chain("How can I update my contact information?"))
    
if __name__ == "__main__":
    create_vector_db()