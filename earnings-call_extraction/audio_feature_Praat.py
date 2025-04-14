import os
import pickle
from tqdm import tqdm
import parselmouth

def measure_pitch(sound, f0min, f0max, unit):
    """Extract 18 features using Praat."""
    try:
        pitch = parselmouth.praat.call(sound, "To Pitch", 0.0, f0min, f0max)
        meanF0 = parselmouth.praat.call(pitch, "Get mean", 0, 0, unit)
        stdevF0 = parselmouth.praat.call(pitch, "Get standard deviation", 0, 0, unit)
        harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        local_jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        local_absolute_jitter = parselmouth.praat.call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        rap_jitter = parselmouth.praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5_jitter = parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        ddp_jitter = parselmouth.praat.call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        local_shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        local_db_shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq3_shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq5_shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq11_shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        dda_shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        n_pulses = parselmouth.praat.call(point_process, "Get number of points")
        n_periods = parselmouth.praat.call(point_process, "Get number of periods", 0.0, 0.0, 0.0001, 0.02, 1.3)
        degree_of_voice_breaks = sum(
            period for period in [
                parselmouth.praat.call(point_process, "Get time from index", i + 1) -
                parselmouth.praat.call(point_process, "Get time from index", i)
                for i in range(1, n_pulses)
            ] if period > 0.02
        ) / sound.duration
        mean_intensity = sound.get_intensity()

        return [
            meanF0, stdevF0, hnr, local_jitter, local_absolute_jitter, rap_jitter, ppq5_jitter, ddp_jitter,
            local_shimmer, local_db_shimmer, apq3_shimmer, apq5_shimmer, apq11_shimmer, dda_shimmer,
            n_pulses, n_periods, degree_of_voice_breaks, mean_intensity
        ]
    except Exception as e:
        return str(e)  # Return the error message if extraction fails

def process_audio_features(folder_path, pickle_path, f0min=75, f0max=500, unit="Hertz"):
    """Process audio features and store them in a dictionary."""
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                audio_feat_dict = pickle.load(f)
        except (EOFError, FileNotFoundError):
            audio_feat_dict = {}
    else:
        audio_feat_dict = {}

    missing_features = {i: [] for i in range(18)}  # Track missing features by index

    for company in tqdm(os.listdir(folder_path)):
        company_path = os.path.join(folder_path, company, "CEO")
        if company in audio_feat_dict:
            continue

        audio_feat_dict[company] = {}
        for audio_file in os.listdir(company_path):
            audio_path = os.path.join(company_path, audio_file)
            try:
                sound = parselmouth.Sound(audio_path)
                features = measure_pitch(sound, f0min, f0max, unit)

                # Handle errors during feature extraction
                if isinstance(features, str):  # If `measure_pitch` returned an error message
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
pickle_path = 'audio-feature_extraction/pklFiles/audio_featDict.pkl'
audio_feat_dict, missing_features = process_audio_features(folder_path, pickle_path)

# Report missing features
for idx, files in missing_features.items():
    if files:
        print(f"Feature {idx + 1} missing in {len(files)} files.")
