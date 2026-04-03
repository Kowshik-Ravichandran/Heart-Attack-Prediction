# ============================================================
# download_new_ecg_data.py (Updated)
# ============================================================
# Downloads REAL ECG data from OTHER PhysioNet databases
# that the model has NEVER seen during training.
#
# Forces the script to dig deeper into the records to find
# TRUE ABNORMAL beats (PVCs, etc.) from different hospitals.
# ============================================================

import wfdb
import numpy as np
import os

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "new_unseen_test_data"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 100  # 100 before + 100 after R-peak = 200 total

normal_count = 0
abnormal_count = 0


def save_beat(beat, label_char, label_name, source_db, rec_id):
    """Save a single heartbeat as CSV and NPY."""
    global normal_count, abnormal_count

    if label_name == "normal":
        normal_count += 1
        count = normal_count
    else:
        abnormal_count += 1
        count = abnormal_count

    filename_base = f"unseen_{label_name}_{count}_{source_db}_{rec_id}"

    np.savetxt(
        os.path.join(OUTPUT_DIR, f"{filename_base}.csv"),
        beat, delimiter=",", fmt="%.6f"
    )
    np.save(
        os.path.join(OUTPUT_DIR, f"{filename_base}.npy"),
        beat
    )
    print(f"   ✅ {label_name.capitalize()} beat ('{label_char}') → {filename_base}.csv")


def extract_from_physionet(pn_dir, rec_id, ann_ext='atr', max_normal=2, max_abnormal=2):
    """
    Stream a record directly from PhysioNet and extract heartbeat segments.
    Explicitly hunts for both NORMAL and ABNORMAL beats.
    """
    try:
        # First, read the header to find the actual signal length
        header = wfdb.rdheader(rec_id, pn_dir=pn_dir)
        actual_len = header.sig_len
        sanitized_sampto = min(300000, actual_len)
        
        # Read a larger chunk to ensure we find abnormal beats
        record = wfdb.rdrecord(rec_id, pn_dir=pn_dir, sampto=sanitized_sampto)
        annotation = wfdb.rdann(rec_id, ann_ext, pn_dir=pn_dir, sampto=sanitized_sampto)

        signal = record.p_signal[:, 0]
        r_peaks = annotation.sample
        labels = annotation.symbol

        n_count = 0
        a_count = 0

        for i in range(1, len(r_peaks) - 1):
            if n_count >= max_normal and a_count >= max_abnormal:
                break

            start = r_peaks[i] - WINDOW_SIZE
            end = r_peaks[i] + WINDOW_SIZE

            if start > 0 and end < len(signal) and (end - start) == 200:
                beat = signal[start:end]

                # Label 'N' is normal. 'V', 'F', 'S', etc. are usually abnormal arrhythmias.
                # In QTDB, 'N', 'p', 't', '(', ')' are regular landmarks.
                is_normal = (labels[i] in ['N', 'p', 't', '(', ')'])

                if is_normal and n_count < max_normal:
                    save_beat(beat, labels[i], "normal", pn_dir.replace('/', '_'), rec_id)
                    n_count += 1
                elif not is_normal and a_count < max_abnormal and labels[i] not in ['+', '~', '|', 'Q', '?']:
                    # Exclude non-beat annotations like rhythm changes (+) or signal quality (~)
                    save_beat(beat, labels[i], "abnormal", pn_dir.replace('/', '_'), rec_id)
                    a_count += 1

        return True
    except Exception as e:
        print(f"   ❌ Error with {pn_dir}/{rec_id}: {e}")
        return False


print("=" * 60)
print("  Downloading NEW ECG Data (Unseen by Model)")
print("  Hunting for BOTH Normal and Abnormal beats...")
print("=" * 60)
print()

# ────────────────────────────────────────────────────────────
# SOURCE 1: QT Database (qtdb)
# ────────────────────────────────────────────────────────────
print("📥 Source 1: QT Database (Different patients & equipment)")
print()

qt_records = ['sel100', 'sel102', 'sel104', 'sel114', 'sel116']
for rec in qt_records:
    print(f"   ⏳ Streaming record {rec}...")
    extract_from_physionet('qtdb', rec, ann_ext='pu1', max_normal=1, max_abnormal=2)

print()

# ────────────────────────────────────────────────────────────
# SOURCE 2: European ST-T Database (edb)
# ────────────────────────────────────────────────────────────
print("📥 Source 2: European ST-T Database (Different hospital)")
print()

edb_records = ['e0103', 'e0104', 'e0105', 'e0106']
for rec in edb_records:
    print(f"   ⏳ Streaming record {rec}...")
    extract_from_physionet('edb', rec, ann_ext='atr', max_normal=1, max_abnormal=2)

print()

# ────────────────────────────────────────────────────────────
# SOURCE 3: St Petersburg INCART 12-lead Arrhythmia Database
# ────────────────────────────────────────────────────────────
print("📥 Source 3: St Petersburg INCART Arrhythmia Database")
print()

incart_records = ['I01', 'I02', 'I03', 'I04']
for rec in incart_records:
    print(f"   ⏳ Streaming record {rec}...")
    extract_from_physionet('incartdb', rec, ann_ext='atr', max_normal=1, max_abnormal=2)

print()

# ────────────────────────────────────────────────────────────
# SUMMARY
# ────────────────────────────────────────────────────────────
print("=" * 60)
print(f"  ✅ Hunt Complete!")
print(f"  📂 Output: {OUTPUT_DIR}")
print(f"  🟢 Normal samples:   {normal_count}")
print(f"  🔴 Abnormal samples: {abnormal_count}")
print(f"  📊 Total samples:    {normal_count + abnormal_count}")
print("=" * 60)
print()
print("  ⭐ These are from COMPLETELY DIFFERENT databases!")
print("     Test these UNSEEN beats in the Streamlit app:")
print("     http://localhost:8501")
