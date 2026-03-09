import numpy as np
import soundfile as sf
import os

FS      = 512
CLIPS   = "test_clips"

conversions = [
    (
        r"test_clips\aQ_00002_sess12_train.npy",
        r"test_clips\ambient_Q_00002_sess12.wav",
        "Ambient — no vessel"
    ),
    (
        r"test_clips\v_0001_0046080_train.npy",
        r"test_clips\vessel_event001_offset90s.wav",
        "Vessel event 1 — 30s window"
    ),
]

for npy_path, wav_path, desc in conversions:
    if not os.path.exists(npy_path):
        print(f"NOT FOUND: {npy_path}")
        continue

    tensor = np.load(npy_path).flatten().astype(np.float32)
    peak   = np.abs(tensor).max()
    audio  = tensor / (peak + 1e-9) * 0.5

    sf.write(wav_path, audio, FS, subtype='PCM_16')
    print(f"{desc}")
    print(f"  {npy_path}  →  {wav_path}")
    print(f"  Duration: {len(tensor)/FS:.1f} s   RMS: {np.sqrt(np.mean(tensor**2)):.6f}")

print("\nDone.")