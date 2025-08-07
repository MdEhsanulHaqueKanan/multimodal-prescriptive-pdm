import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
import app_config as config
from huggingface_hub import InferenceClient # <-- NEW, DIRECT IMPORT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_rag_chain():
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
        
        # Create a client to talk to the Inference API directly
        client = InferenceClient(model="google/flan-t5-large")

        def llm_call(prompt_text: str) -> str:
            """ A simple function to call the HF API. """
            response = client.text_generation(prompt=prompt_text, max_new_tokens=512)
            return response

        # Wrap our function in a RunnableLambda to use it in the chain
        llm = RunnableLambda(llm_call)

    else:
        # Local development code remains the same
        logging.info("Initializing LLM for local Ollama development.")
        llm_model_name = os.getenv("LLM_MODEL_NAME", "llama3:8b")
        base_url = "http://host.docker.internal:11434" if os.getenv("DOCKER_ENV") else "http://localhost:11434"
        llm = Ollama(model=llm_model_name, base_url=base_url)
    
    # The chain structure remains the same
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("RAG chain created successfully.")
    return rag_chain

# ... (keep your if __name__ == '__main__': block) ...