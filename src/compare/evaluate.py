"""
evaluate.py - score a trained model on val or test, the same way for every model.

For each clip: run the model on the noisy magnitude, multiply by the mask, put the
NOISY phase back, inverse-STFT to a wav, then compare against the clean wav.

Metrics (noisy input -> denoised output):
  SNR   waveform signal-to-noise ratio in dB: 10*log10(sum clean^2 / sum (clean - x)^2).
        For the noisy input, clean - x is exactly the added noise. The CA's one hard rule:
        this must go UP. We also count how many clips got WORSE than their input.
  PESQ  wideband PESQ (ITU-T P.862.2), roughly 1.0 (bad) .. 4.6 (perfect). Should go up.
  STOI  0 .. 1 intelligibility. Calculated, but the CA said not to focus on it.

Results are broken down by noise type and by input SNR level (-5 .. 15 dB), because
averages hid the fact that the first LSTM hurt the cleaner clips.
"""
import csv, os, sys
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from features import stft, istft  # noqa: E402  (same STFT as the features)


def wav_snr(clean, est):
    return float(10 * np.log10(np.sum(clean ** 2) / (np.sum((clean - est) ** 2) + 1e-12) + 1e-12))


def pesq_stoi(clean, est, sr):
    from pesq import pesq
    from pystoi import stoi
    try:
        p = float(pesq(sr, clean, est, "wb"))
    except Exception:
        p = float("nan")
    return p, float(stoi(clean, est, sr, extended=False))


def read_manifest(data_dir):
    with open(os.path.join(data_dir, "manifest.csv"), newline="") as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}


@torch.no_grad()
def predict_masks(model, noisy_mag, device, batch=8):
    model.eval()
    out = []
    for s in range(0, len(noisy_mag), batch):
        out.append(model(torch.from_numpy(noisy_mag[s:s + batch]).to(device)).cpu().numpy())
    return np.concatenate(out)


def evaluate(model, split, cfg, device, out_dir=None, save_wavs=0):
    """Returns (rows, summary). Writes per-clip CSV and optional denoised wavs to out_dir."""
    feats = np.load(os.path.join(cfg["feat_dir"], f"{split}.npz"))
    noisy_mag, ids = feats["noisy_mag"], [str(p) for p in feats["pair_id"]]
    masks = predict_masks(model, noisy_mag, device)
    manifest = read_manifest(cfg["data_dir"])
    sr = cfg["sr"]

    rows = []
    for i, pid in enumerate(ids):
        d = os.path.join(cfg["data_dir"], split)
        noisy, _ = sf.read(os.path.join(d, "noisy", pid + ".wav"), dtype="float32")
        clean, _ = sf.read(os.path.join(d, "clean", pid + ".wav"), dtype="float32")
        est = istft(masks[i] * stft(noisy), length=len(noisy)).astype(np.float32)
        p_in, s_in = pesq_stoi(clean, noisy, sr)
        p_out, s_out = pesq_stoi(clean, est, sr)
        m = manifest[pid]
        rows.append(dict(
            pair_id=pid, noise_type=m["noise_type"], snr_level=int(m["snr_db"]),
            snr_in=wav_snr(clean, noisy), snr_out=wav_snr(clean, est),
            pesq_in=p_in, pesq_out=p_out, stoi_in=s_in, stoi_out=s_out,
        ))
        if out_dir and i < save_wavs:
            os.makedirs(os.path.join(out_dir, "denoised_wavs"), exist_ok=True)
            sf.write(os.path.join(out_dir, "denoised_wavs", f"{pid}.wav"), est, sr)

    summary = summarise(rows)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{split}_per_clip.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows, summary


def _agg(group):
    def mean(k):
        return float(np.nanmean([r[k] for r in group]))
    return dict(
        n=len(group),
        snr_in=mean("snr_in"), snr_out=mean("snr_out"), snr_gain=mean("snr_out") - mean("snr_in"),
        pct_snr_worse=100.0 * float(np.mean([r["snr_out"] < r["snr_in"] for r in group])),
        pesq_in=mean("pesq_in"), pesq_out=mean("pesq_out"), pesq_gain=mean("pesq_out") - mean("pesq_in"),
        stoi_in=mean("stoi_in"), stoi_out=mean("stoi_out"), stoi_gain=mean("stoi_out") - mean("stoi_in"),
    )


def summarise(rows):
    by_noise, by_snr = defaultdict(list), defaultdict(list)
    for r in rows:
        by_noise[r["noise_type"]].append(r)
        by_snr[r["snr_level"]].append(r)
    return dict(
        overall=_agg(rows),
        by_noise_type={k: _agg(v) for k, v in sorted(by_noise.items())},
        by_snr_level={str(k): _agg(v) for k, v in sorted(by_snr.items())},
    )


def print_summary(name, split, s):
    o = s["overall"]
    print(f"\n=== {name} on {split} (n={o['n']}) - noisy -> denoised ===")
    print(f"  SNR  {o['snr_in']:6.2f} -> {o['snr_out']:6.2f} dB  ({o['snr_gain']:+.2f})   "
          f"clips worse than input: {o['pct_snr_worse']:.0f}%")
    print(f"  PESQ {o['pesq_in']:6.3f} -> {o['pesq_out']:6.3f}     ({o['pesq_gain']:+.3f})")
    print(f"  STOI {o['stoi_in']:6.3f} -> {o['stoi_out']:6.3f}     ({o['stoi_gain']:+.3f})")
    for title, key in (("by input SNR level", "by_snr_level"), ("by noise type", "by_noise_type")):
        print(f"  {title}:")
        for k, g in s[key].items():
            print(f"    {k:>9}  SNR {g['snr_in']:6.2f} -> {g['snr_out']:6.2f}  "
                  f"PESQ {g['pesq_in']:.2f} -> {g['pesq_out']:.2f}  STOI {g['stoi_in']:.3f} -> {g['stoi_out']:.3f}")
