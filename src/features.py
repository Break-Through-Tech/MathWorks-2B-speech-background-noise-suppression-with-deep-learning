"""
features.py - turn the wav pairs from make_pairs.py into STFT magnitude spectrograms.

    python src/features.py

Writes data/processed/features/{split}.npz with
    noisy_mag, clean_mag : float32 [N, 257, 376]   (freq bins x frames)
    pair_id              : [N]
Phase is NOT stored: at inference you take the noisy wav's own phase (recompute with stft()).
Log-compression (e.g. np.log1p) is left to the model code so everyone shares the same raw features.
"""
import os, csv
import numpy as np
import soundfile as sf
import librosa

SR, N_FFT, HOP, WIN = 16_000, 512, 128, "hann"   # 32 ms window, 8 ms hop -> 257 freq bins


def stft(x):
    return librosa.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=WIN, center=True)


def istft(S, length):
    return librosa.istft(S, hop_length=HOP, win_length=N_FFT, window=WIN, center=True, length=length)


def main(root="data/processed"):
    rows = list(csv.DictReader(open(os.path.join(root, "manifest.csv"))))
    os.makedirs(os.path.join(root, "features"), exist_ok=True)
    for split in ("train", "val", "test"):
        ids = [r["pair_id"] for r in rows if r["split"] == split]
        noisy, clean = [], []
        for pid in ids:
            xn, _ = sf.read(os.path.join(root, split, "noisy", pid + ".wav"), dtype="float32")
            xc, _ = sf.read(os.path.join(root, split, "clean", pid + ".wav"), dtype="float32")
            noisy.append(np.abs(stft(xn))); clean.append(np.abs(stft(xc)))
        noisy, clean = np.stack(noisy), np.stack(clean)
        np.savez(os.path.join(root, "features", f"{split}.npz"),
                 noisy_mag=noisy, clean_mag=clean, pair_id=np.array(ids))
        print(split, noisy.shape, f"{noisy.nbytes * 2 / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
