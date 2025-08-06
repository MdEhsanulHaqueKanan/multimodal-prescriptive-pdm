import os
import logging
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import app_config as config
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

KNOWLEDGE_BASE_FILES = {
    "manual": {"path": os.path.join(config.RAW_DATA_DIR, "turbofan_service_manual.md"), "loader": TextLoader},
    "history": {"path": os.path.join(config.RAW_DATA_DIR, "maintenance_history.csv"), "loader": CSVLoader, "loader_kwargs": {"source_column": "notes"}}
}

def load_knowledge_base_documents():
    all_documents = []
    for doc_info in KNOWLEDGE_BASE_FILES.values():
        if os.path.exists(doc_info["path"]):
            LoaderClass = doc_info["loader"]
            loader_kwargs = doc_info.get("loader_kwargs", {})
            if LoaderClass == TextLoader: loader_kwargs["encoding"] = "utf-8"
            loader = LoaderClass(doc_info["path"], **loader_kwargs)
            all_documents.extend(loader.load())
    return all_documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def build_and_save_vector_store(chunks, db_path):
    # This function now correctly expects a string for db_path
    embedding_model = HuggingFaceEmbeddings(model_name="./embedding_model")
    Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=db_path)

def run_ingestion_if_needed():
    # Convert the Path object to a string here for all downstream use
    db_path_str = str(config.DB_PATH)
    
    if os.path.exists(db_path_str):
        logging.info(f"Vector store found at {db_path_str}. Skipping ingestion.")
        return
        
    logging.info("Vector store not found. Starting data ingestion process...")
    try:
        loaded_docs = load_knowledge_base_documents()
        if loaded_docs:
            doc_chunks = split_documents(loaded_docs)
            # Pass the string path to the function
            build_and_save_vector_store(chunks=doc_chunks, db_path=db_path_str)
            logging.info("--- Knowledge base ingestion complete. ---")
    except Exception as e:
        logging.error(f"Error during initial data ingestion: {e}", exc_info=True)

if __name__ == "__main__":
    run_ingestion_if_needed()