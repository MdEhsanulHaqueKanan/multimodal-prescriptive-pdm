import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
import app_config as config
from huggingface_hub import InferenceClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_rag_chain():
    """
    Creates and returns a complete RAG chain.
    """
    logging.info("Creating the RAG chain...")
    embedding_model = HuggingFaceEmbeddings(model_name="./embedding_model")
    vector_store = Chroma(persist_directory=str(config.DB_PATH), embedding_function=embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt_template_str = """
    Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = PromptTemplate.from_template(prompt_template_str)

    deployment_platform = os.getenv("DEPLOYMENT_PLATFORM", "local")
    
    if deployment_platform == "huggingface":
        logging.info("Initializing LLM for Hugging Face deployment using direct InferenceClient.")
        
        try:
            # Read the token from the environment secret
            token = os.getenv("HF_TOKEN")
            client = InferenceClient(model="google/flan-t5-large", token=token)

            def llm_call(input_dict: dict) -> str:
                """ A custom function to format the prompt and call the HF API. """
                try:
                    # Manually format the prompt using the retrieved context and question
                    formatted_prompt = prompt.format(
                        context=input_dict["context"], 
                        question=input_dict["question"]
                    )
                    # Directly call the client's text_generation method
                    response = client.text_generation(prompt=formatted_prompt, max_new_tokens=512)
                    return response
                except Exception as e:
                    logging.error(f"Error during direct API call: {e}")
                    return "Error: Could not get a response from the language model."

            # The entire LLM part of the chain is now our custom function
            # We use StrOutputParser to ensure the final output is a clean string
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | RunnableLambda(llm_call)
                | StrOutputParser()
            )

        except Exception as e:
            logging.error(f"Failed to initialize InferenceClient: {e}")
            return None # Return None if initialization fails

    else:
        # Local Ollama development path remains the same
        logging.info("Initializing LLM for local Ollama development.")
        llm_model_name = os.getenv("LLM_MODEL_NAME", "llama3:8b")
        base_url = "http://host.docker.internal:11434" if os.getenv("DOCKER_ENV") else "http://localhost:11434"
        llm = Ollama(model=llm_model_name, base_url=base_url)
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
        test_question = "How do I replace the bearing assembly?"
        print(f"\n[?] Test Question: {test_question}")
        answer = chain.invoke(test_question)
        print("\n[!] AI-Generated Answer:")
        print(answer)
        print("\n--- RAG Chain test complete. ---")
    except Exception as e:
        logging.error(f"An error occurred during the RAG chain test: {e}", exc_info=True)