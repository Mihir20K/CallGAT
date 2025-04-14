import os
import pickle
import librosa
import numpy as np
from tqdm import tqdm

def get_mfcc_features(y, sr, n_mfcc=13):
    """Extracts MFCC features (mean and std for each coefficient)."""
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfccs, axis=1)  # Mean of each MFCC
        mfcc_std = np.std(mfccs, axis=1)    # Std of each MFCC
        return np.concatenate([mfcc_mean, mfcc_std])  # Combine mean and std
    except Exception as e:
        return str(e)

def process_audio_features(folder_path, pickle_path, n_mfcc=13):
    """Processes audio features and stores them in a dictionary."""
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                audio_feat_dict = pickle.load(f)
        except (EOFError, FileNotFoundError):
            audio_feat_dict = {}
    else:
        audio_feat_dict = {}

    missing_features = {i: [] for i in range(n_mfcc * 2)}  # Track missing features

    for company in tqdm(os.listdir(folder_path)):
        company_path = os.path.join(folder_path, company, "CEO")
        if company in audio_feat_dict:
            continue

        audio_feat_dict[company] = {}
        for audio_file in os.listdir(company_path):
            audio_path = os.path.join(company_path, audio_file)
            try:
                y, sr = librosa.load(audio_path, sr=None)
                features = get_mfcc_features(y, sr, n_mfcc)

                # Handle errors during feature extraction
                if isinstance(features, str):
                    print(f"Error in file {audio_file}: {features}")
                    continue

                # Check for missing features and log them
                for idx, value in enumerate(features):
                    if value is None or isinstance(value, str):
                        missing_features[idx].append(f"{company}/{audio_file}")

                audio_feat_dict[company][audio_file[:-4]] = features
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

        # Save progress to pickle
        with open(pickle_path, 'wb') as f:
            pickle.dump(audio_feat_dict, f)

    return audio_feat_dict, missing_features

# Usage
folder_path = './OGdataset'
pickle_path = './audio_featDict_MFCC.pkl'
audio_feat_dict, missing_features = process_audio_features(folder_path, pickle_path)

# Report missing features
for idx, files in missing_features.items():
    if files:
        print(f"MFCC Feature {idx + 1} missing in {len(files)} files.")
