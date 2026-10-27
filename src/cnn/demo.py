"""
demo.py - denoise one test clip with the selected mask_ch32 checkpoint.

    .venv/bin/python src/cnn/demo.py
    .venv/bin/python src/cnn/demo.py --id test_0003

Writes demo/<id>_noisy.wav, demo/<id>_denoised.wav, demo/<id>_clean.wav,
and demo/<id>_spectrograms.png.
"""
import argparse, os, sys
from pathlib import Path

CNN = Path(__file__).resolve().parent
SRC = CNN.parent
ROOT = SRC.parent
if __name__ == "__main__":
    _venv_python = ROOT / ".venv" / "bin" / "python"
    if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
        os.execv(str(_venv_python), [str(_venv_python), *sys.argv])

for _path in (SRC, CNN):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from features import stft
from train import SR, load_checkpoint, pick_device, predict_magnitudes, read_wav, reconstruct


def main():
    parser = argparse.ArgumentParser(description="Denoise one test clip for a demo")
    parser.add_argument("--id", default="test_0000", help="pair id in the test split, e.g. test_0000")
    args = parser.parse_args()
    pid = args.id

    noisy_path = ROOT / "dataset/audio/test/noisy" / f"{pid}.wav"
    clean_path = ROOT / "dataset/audio/test/clean" / f"{pid}.wav"
    checkpoint = ROOT / "runs/mask_ch32/best.pt"
    for path in (noisy_path, clean_path, checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing {path}")

    noisy_wav = read_wav(noisy_path)
    clean_wav = read_wav(clean_path)
    noisy_mag = np.abs(stft(noisy_wav)).astype("float32")

    device = pick_device()
    model, ckpt = load_checkpoint(checkpoint, device)
    pred_mag = predict_magnitudes(model, noisy_mag[None], ckpt["mean"], ckpt["std"], 1, device)[0]
    enhanced = reconstruct(pred_mag, noisy_wav)

    demo = ROOT / "demo"
    demo.mkdir(exist_ok=True)
    written = {
        "noisy": demo / f"{pid}_noisy.wav",
        "denoised": demo / f"{pid}_denoised.wav",
        "clean": demo / f"{pid}_clean.wav",
        "figure": demo / f"{pid}_spectrograms.png",
    }
    sf.write(written["noisy"], noisy_wav, SR)
    sf.write(written["denoised"], enhanced, SR)
    sf.write(written["clean"], clean_wav, SR)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    clean_mag = np.abs(stft(clean_wav)).astype("float32")
    for ax, mag, title in zip(axes, (noisy_mag, pred_mag, clean_mag), ("noisy", "model", "clean")):
        ax.imshow(np.log1p(np.maximum(mag, 0)), origin="lower", aspect="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("frame")
    axes[0].set_ylabel("frequency bin")
    fig.suptitle(pid)
    fig.tight_layout()
    fig.savefig(written["figure"], dpi=150, bbox_inches="tight")

    print(f"{pid} on {device}")
    for label, path in written.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
