from sentence_transformers import SentenceTransformer

# Define the model name and the local path to save it to
model_name = 'all-MiniLM-L6-v2'
save_path = './embedding_model'

print(f"Downloading model '{model_name}' to '{save_path}'...")

# Download the model from Hugging Face Hub and save it to the specified path
model = SentenceTransformer(model_name)
model.save(save_path)

print("Model downloaded and saved successfully.")