# ============================================================
# generate_sample_data.py — Create Sample ECG Test Files
# ============================================================
# Run this script to generate sample .csv and .npy files
# that can be used to test the Streamlit app.
#
# These samples simulate a synthetic ECG-like waveform
# with 200 data points (matching the model's expected input).
# ============================================================

import numpy as np
import os

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
os.makedirs(output_dir, exist_ok=True)


def generate_synthetic_ecg(num_samples=200, seed=42):
    """
    Generate a simple synthetic ECG-like signal.
    This is NOT a real ECG — it's just for testing the app.
    """
    np.random.seed(seed)
    t = np.linspace(0, 1, num_samples)

    # Simulate P-wave, QRS complex, and T-wave
    p_wave = 0.1 * np.sin(2 * np.pi * 5 * t) * np.exp(-((t - 0.15) ** 2) / 0.002)
    qrs = 0.8 * np.sin(2 * np.pi * 12 * t) * np.exp(-((t - 0.45) ** 2) / 0.001)
    t_wave = 0.2 * np.sin(2 * np.pi * 3 * t) * np.exp(-((t - 0.7) ** 2) / 0.005)

    # Combine components with some noise
    signal = p_wave + qrs + t_wave + 0.02 * np.random.randn(num_samples)
    return signal


# ── Generate Normal-like sample ──
normal_signal = generate_synthetic_ecg(seed=42)
np.save(os.path.join(output_dir, "sample_normal.npy"), normal_signal)
np.savetxt(
    os.path.join(output_dir, "sample_normal.csv"),
    normal_signal,
    delimiter=",",
    fmt="%.6f",
)

# ── Generate another sample with different characteristics ──
abnormal_signal = generate_synthetic_ecg(seed=99)
# Add some irregularities to simulate abnormal pattern
abnormal_signal[80:120] += 0.5 * np.random.randn(40)
np.save(os.path.join(output_dir, "sample_abnormal.npy"), abnormal_signal)
np.savetxt(
    os.path.join(output_dir, "sample_abnormal.csv"),
    abnormal_signal,
    delimiter=",",
    fmt="%.6f",
)

print("✅ Sample data generated in:", output_dir)
print("   - sample_normal.csv / .npy")
print("   - sample_abnormal.csv / .npy")
print("\nUse these files to test the Streamlit app.")
