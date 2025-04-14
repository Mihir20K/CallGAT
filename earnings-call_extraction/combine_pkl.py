import os
import pickle
import numpy as np
from tqdm import tqdm

# Path to the directory containing individual .pkl files
pkl_dir = r"C:\Users\mihir\Desktop\Stock_Volatility\CallGAP\audio-feature_extraction\finbert_embeddings"

# Path to save the combined .pkl file
combined_pkl_path = os.path.join(pkl_dir, "combined_finbert_embeddings.pkl")

# Dictionary to hold combined data
combined_embeddings = {}

# Iterate through each .pkl file and merge into the combined dictionary
for pkl_file in tqdm(os.listdir(pkl_dir), desc="Combining embeddings"):
    if pkl_file.endswith(".pkl") and pkl_file != "combined_finbert_embeddings.pkl":
        company_name = os.path.splitext(pkl_file)[0]  # Extract company name
        pkl_path = os.path.join(pkl_dir, pkl_file)

        with open(pkl_path, "rb") as f:
            embeddings = pickle.load(f)
            combined_embeddings[company_name] = np.array(embeddings)

# Save the combined embeddings as a single .pkl file
with open(combined_pkl_path, "wb") as f:
    pickle.dump(combined_embeddings, f, protocol=4)

print(f"Combined embeddings saved successfully at: {combined_pkl_path}")
