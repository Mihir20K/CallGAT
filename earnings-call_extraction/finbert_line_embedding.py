import os
import pickle
import gc
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the FinBERT tokenizer and model in optimized mode
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModel.from_pretrained("ProsusAI/finbert").to(device).eval()

# Define dataset path
og_dataset_path = r"C:\Users\mihir\Desktop\Stock_Volatility\CallGAP\audio-feature_extraction\OGdataset"

# Define pickle save path
pkl_dir = "finbert_embeddings"
os.makedirs(pkl_dir, exist_ok=True)  # Create folder to store embeddings

# Function to generate embeddings
def generate_embeddings(text):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
    ).to(device)  # Ensure tensors are on GPU

    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()  # Move output back to CPU

# Get list of already processed companies
processed_companies = set(f.split(".")[0] for f in os.listdir(pkl_dir))

# Process each company's transcript
for company_folder in tqdm(os.listdir(og_dataset_path), desc="Processing companies"):
    if company_folder in processed_companies:
        continue  # Skip already processed companies

    company_path = os.path.join(og_dataset_path, company_folder)
    transcript_file = os.path.join(company_path, "TextSequence.txt")

    if os.path.isdir(company_path) and os.path.isfile(transcript_file):
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript_lines = f.readlines()

        company_embeddings = []

        for idx, line in enumerate(transcript_lines):
            cleaned_line = line.strip()
            if cleaned_line:
                try:
                    embeddings = generate_embeddings(cleaned_line)
                    company_embeddings.append(embeddings)
                except Exception as e:
                    print(f"Error generating embedding for {company_folder} - Line {idx + 1}: {e}")
                    company_embeddings.append(np.zeros(768))  # Use NumPy for efficiency

        # Save each company's embeddings separately
        pkl_path = os.path.join(pkl_dir, f"{company_folder}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(company_embeddings, f, protocol=4)

        # Free memory
        del company_embeddings
        gc.collect()  # Force garbage collection
        torch.cuda.empty_cache()  # Clear unused GPU memory

print("Line-wise embeddings successfully generated and stored in separate files.")
