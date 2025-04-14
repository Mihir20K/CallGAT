import os
import pickle
from tqdm import tqdm
import librosa
import parselmouth

def get_prosodic_features(sound, y, sr):
    """Extracts 8 prosodic features from an audio file."""
    try:
        # Compute energy features
        energy = librosa.feature.rms(y=y)
        SD_energy = energy.std()  # Standard deviation of energy
        
        # Compute pitch features
        pitch = parselmouth.praat.call(sound, "To Pitch", 0.0, 75, 300)
        maxPitch = parselmouth.praat.call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        minPitch = parselmouth.praat.call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        
        # Compute intensity features
        intensity = parselmouth.praat.call(sound, "To Intensity", 75, 0)
        maxIntensity = parselmouth.praat.call(intensity, "Get maximum", 0, 0, "Parabolic")
        minIntensity = parselmouth.praat.call(intensity, "Get minimum", 0, 0, "Parabolic")
        
        # Compute voiced/unvoiced frame ratio
        voiced_frames = pitch.count_voiced_frames()
        total_frames = pitch.get_number_of_frames()
        voiced_to_total_ratio = voiced_frames / total_frames if total_frames > 0 else 0
        voiced_to_unvoiced_ratio = voiced_frames / (total_frames - voiced_frames) if total_frames > voiced_frames else 0
        
        return [
            SD_energy, maxIntensity, minIntensity, maxPitch, minPitch,
            voiced_frames, voiced_to_total_ratio, voiced_to_unvoiced_ratio
        ]
    except Exception as e:
        return str(e)  # Return the error message if extraction fails

def process_audio_features(folder_path, pickle_path):
    """Processes audio features and stores them in a dictionary."""
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                audio_feat_dict = pickle.load(f)
        except (EOFError, FileNotFoundError):
            audio_feat_dict = {}
    else:
        audio_feat_dict = {}

    missing_features = {i: [] for i in range(8)}  # Track missing features by index

    for company in tqdm(os.listdir(folder_path)):
        company_path = os.path.join(folder_path, company, "CEO")
        if company in audio_feat_dict:
            continue

        audio_feat_dict[company] = {}
        for audio_file in os.listdir(company_path):
            audio_path = os.path.join(company_path, audio_file)
            try:
                sound = parselmouth.Sound(audio_path)
                y, sr = librosa.load(audio_path, sr=None)
                features = get_prosodic_features(sound, y, sr)

                # Handle errors during feature extraction
                if isinstance(features, str):  # If `get_prosodic_features` returned an error message
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
folder_path = 'audio-feature_extraction/OGdataset'
pickle_path = 'audio-feature_extraction/pklFiles/audio_featDictMark2.pkl'
audio_feat_dict, missing_features = process_audio_features(folder_path, pickle_path)

# Report missing features
for idx, files in missing_features.items():
    if files:
        print(f"Feature {idx + 1} missing in {len(files)} files.")
