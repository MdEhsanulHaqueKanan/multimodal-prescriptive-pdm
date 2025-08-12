import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import app_config as config

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_rag_chain():
    """
    Creates and returns the RAG chain configured for local development.
    """
    logging.info("Creating the RAG chain for local execution...")

    # 1. Initialize the components
    embedding_model = HuggingFaceEmbeddings(model_name="./embedding_model")
    
    vector_store = Chroma(
        persist_directory=str(config.DB_PATH), 
        embedding_function=embedding_model
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # --- Use Local Ollama Server ---
    # This connects to the Ollama app running on your main PC.
    llm_model_name = os.getenv("LLM_MODEL_NAME", "llama3:8b")
    base_url = "http://localhost:11434"
    
    try:
        llm = Ollama(model=llm_model_name, base_url=base_url)
        logging.info(f"Connecting to local Ollama with model: {llm_model_name} at {base_url}")
    except Exception as e:
        logging.error(f"Failed to initialize Ollama: {e}", exc_info=True)
        return None

    # 2. Define the Prompt Template
    prompt_template_str = """
    You are an expert maintenance assistant. Your task is to provide clear, accurate, and helpful answers based exclusively on the following context.
    If the context does not contain the answer to the question, state clearly that you cannot answer with the provided information. 
    Do not make up information or use any external knowledge.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = PromptTemplate.from_template(prompt_template_str)

    # 3. Build the RAG Chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("RAG chain created successfully.")
    return rag_chain

# --- Execution Block for Testing ---
if __name__ == '__main__':
    print("--- Testing the RAG Chain (Local Ollama Mode) ---")
    try:
        chain = create_rag_chain()
        if chain:
            test_question = "How do I replace the bearing assembly?"
            print(f"\n[?] Test Question: {test_question}")
            answer = chain.invoke(test_question)
            print("\n[!] AI-Generated Answer:")
            print(answer)
            print("\n--- RAG Chain test complete. ---")
        else:
            print("Chain initialization failed.")
    except Exception as e:
        logging.error(f"An error occurred during the RAG chain test: {e}", exc_info=True)