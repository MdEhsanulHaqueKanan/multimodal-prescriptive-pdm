import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import app_config as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_rag_chain():
    """
    Creates and returns a complete RAG chain.
    """
    logging.info("Creating the RAG chain...")
    embedding_model = HuggingFaceEmbeddings(model_name="./embedding_model")
    vector_store = Chroma(persist_directory=str(config.DB_PATH), embedding_function=embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    deployment_platform = os.getenv("DEPLOYMENT_PLATFORM", "local")
    
    if deployment_platform == "huggingface":
        logging.info("Initializing LLM for Hugging Face deployment.")
        # Use the most basic, universally supported text-generation model for maximum compatibility
        repo_id = "gpt2" 
        try:
            llm = HuggingFaceEndpoint(
                repo_id=repo_id,
                max_new_tokens=256 # Use a slightly shorter length for this classic model
            )
            logging.info(f"Using Hugging Face Endpoint with model: {repo_id}")
        except Exception as e:
            logging.error(f"Failed to initialize HuggingFaceEndpoint: {e}", exc_info=True)
            return None
    else:
        # Local development code remains the same
        logging.info("Initializing LLM for local Ollama development.")
        llm_model_name = os.getenv("LLM_MODEL_NAME", "llama3:8b")
        base_url = "http://host.docker.internal:11434" if os.getenv("DOCKER_ENV") else "http://localhost:11434"
        try:
            llm = Ollama(model=llm_model_name, base_url=base_url)
            logging.info(f"Using Ollama with model: {llm_model_name} at {base_url}")
        except Exception as e:
            logging.error(f"Failed to initialize Ollama: {e}", exc_info=True)
            return None
            
    if llm is None:
        # This case is now handled within the try/except blocks
        raise ValueError("LLM could not be initialized.")

    prompt_template_str = """
    Answer the question based only on the following context.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template_str)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("RAG chain created successfully.")
    return rag_chain

if __name__ == '__main__':
    # This block remains for local testing
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