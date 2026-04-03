# ============================================================
# extract_real_samples.py
# ============================================================
# This script extracts REAL heartbeat segments from the
# MIT-BIH Arrhythmia Database — the same dataset used to
# train the CNN-LSTM model.
#
# It saves individual heartbeat samples as .csv and .npy files
# that you can upload directly to the Streamlit app for testing.
#
# Each sample = 200 data points (100 before + 100 after R-peak)
# ============================================================

import wfdb
import numpy as np
import os

# ── Configuration ──
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "mit-bih-arrhythmia-database-1.0.0"
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "real_test_samples"
)
WINDOW_SIZE = 100  # 100 samples on each side of R-peak = 200 total

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Records to extract from ──
# Record 100: mostly normal beats
# Record 200: contains many abnormal beats (PVCs)
# Record 208: contains VT and other abnormalities
records_to_extract = {
    "100": "Mostly Normal beats",
    "200": "Contains Abnormal beats (PVCs)",
    "208": "Contains Abnormal beats (VT, PVC)",
    "119": "Contains Abnormal PVCs",
    "220": "All Normal beats",
}

normal_count = 0
abnormal_count = 0

print("=" * 60)
print("  Extracting Real ECG Heartbeat Samples")
print("  from MIT-BIH Arrhythmia Database")
print("=" * 60)
print()

for rec_id, description in records_to_extract.items():
    record_path = os.path.join(DATASET_PATH, rec_id)

    print(f"📁 Processing Record {rec_id}: {description}")

    try:
        # Load the ECG signal and annotations
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')

        signal = record.p_signal[:, 0]  # Use channel 0 (MLII)
        r_peaks = annotation.sample
        labels = annotation.symbol

        # Extract up to 5 beats per record
        beats_extracted = 0

        for i in range(1, len(r_peaks) - 1):
            if beats_extracted >= 5:
                break

            start = r_peaks[i] - WINDOW_SIZE
            end = r_peaks[i] + WINDOW_SIZE

            # Make sure we don't go out of bounds
            if start > 0 and end < len(signal):
                beat = signal[start:end]

                # Determine if Normal (N) or Abnormal (anything else)
                if labels[i] == 'N':
                    label_name = "normal"
                    normal_count += 1
                    count = normal_count
                else:
                    label_name = "abnormal"
                    abnormal_count += 1
                    count = abnormal_count

                # Save as .csv
                csv_filename = f"real_{label_name}_{count}_rec{rec_id}.csv"
                np.savetxt(
                    os.path.join(OUTPUT_DIR, csv_filename),
                    beat, delimiter=",", fmt="%.6f"
                )

                # Save as .npy
                npy_filename = f"real_{label_name}_{count}_rec{rec_id}.npy"
                np.save(os.path.join(OUTPUT_DIR, npy_filename), beat)

                beat_label = labels[i]
                print(f"   ✅ Beat {beats_extracted+1}: '{beat_label}' → {label_name} → {csv_filename}")

                beats_extracted += 1

    except Exception as e:
        print(f"   ❌ Error processing record {rec_id}: {e}")

    print()

print("=" * 60)
print(f"  ✅ Extraction Complete!")
print(f"  📂 Output folder: {OUTPUT_DIR}")
print(f"  🟢 Normal samples:   {normal_count}")
print(f"  🔴 Abnormal samples: {abnormal_count}")
print(f"  📊 Total samples:    {normal_count + abnormal_count}")
print("=" * 60)
print()
print("  You can now upload these files to the Streamlit app!")
print("  → Open http://localhost:8501")
print("  → Upload any .csv or .npy file from the 'real_test_samples' folder")
