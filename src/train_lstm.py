"""
LSTM speech denoiser for the MathWorks 2B pipeline (kevin-preprocessing branch).

Trains directly on the spectrogram features made by src/features.py:
  data/processed/features/{train,val,test}.npz
    noisy_mag, clean_mag : float32 [N, 257, 376]   (n_fft 512, hop 128, Hann)
    pair_id              : e.g. "train_0000"

If data/processed/manifest.csv exists, training files are picked evenly per noise
type and results are broken down by noise type. If the test .wav files exist
(data/processed/test/{noisy,clean}/), the script also writes denoised audio and
computes PESQ / STOI. Otherwise it reports a spectrogram-based score only.

Run from the repo root:
  python src/train_lstm.py                      # 100 training clips (20 per noise type)
  python src/train_lstm.py --train_per_type 24  # 120 clips
"""
import os, csv, json, random, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

CFG = dict(
    feat_dir="data/processed/features",
    data_dir="data/processed",
    out_dir="runs/lstm_v1",
    train_per_type=20,   # x 5 noise types = 100 training clips
    sr=16000,
    n_fft=512,
    hop=128,
    hidden=256,
    layers=2,
    batch_size=8,
    lr=1e-3,
    epochs=80,
    patience=10,         # stop if validation loss hasn't improved for this many epochs
    seed=42,
)


# ---------------------------------------------------------------- data
def load_split(cfg, split):
    d = np.load(os.path.join(cfg["feat_dir"], f"{split}.npz"))
    return d["noisy_mag"], d["clean_mag"], [str(p) for p in d["pair_id"]]


def read_noise_types(cfg):
    """pair_id -> noise_type, from manifest.csv if it exists."""
    path = os.path.join(cfg["data_dir"], "manifest.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["pair_id"].removesuffix(".wav"): r["noise_type"] for r in csv.DictReader(f)}


def pick_train(ids, noise_types, per_type, rng):
    """Indices of the training clips: per_type per noise type when the manifest is
    available, otherwise a random 5 * per_type clips."""
    if not noise_types:
        idx = list(range(len(ids)))
        rng.shuffle(idx)
        print("  (no manifest.csv found - picking clips at random instead of per noise type)")
        return sorted(idx[:5 * per_type])
    by_type = defaultdict(list)
    for i, pid in enumerate(ids):
        by_type[noise_types.get(pid, "unknown")].append(i)
    picked = []
    for t in sorted(by_type):
        idx = by_type[t][:]
        rng.shuffle(idx)
        if len(idx) < per_type:
            print(f"  warning: only {len(idx)} training clips for '{t}'")
        picked += idx[:per_type]
    return sorted(picked)


# ---------------------------------------------------------------- model
class LSTMDenoiser(nn.Module):
    """Unidirectional, so it can later run in streaming / real-time mode."""

    def __init__(self, n_freq=257, hidden=256, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_freq, hidden, num_layers=layers, batch_first=True,
                            dropout=0.2 if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, n_freq)

    def forward(self, mag):                      # mag: (B, 257, frames)
        x = torch.log1p(mag).transpose(1, 2)     # (B, frames, 257)
        h, _ = self.lstm(x)
        return torch.sigmoid(self.fc(h)).transpose(1, 2)   # mask (B, 257, frames)


def loss_fn(est_mag, clean_mag):
    # MSE on power-compressed magnitudes so quiet speech isn't ignored
    return torch.mean(((est_mag + 1e-8) ** 0.3 - (clean_mag + 1e-8) ** 0.3) ** 2)


def run_epoch(model, loader, device, opt=None):
    train = opt is not None
    model.train(train)
    total, count = 0.0, 0
    with torch.set_grad_enabled(train):
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            loss = loss_fn(model(noisy) * noisy, clean)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            total += loss.item() * noisy.size(0)
            count += noisy.size(0)
    return total / max(count, 1)


# ---------------------------------------------------------------- evaluation
def spec_snr(clean, est):
    """Per-clip SNR (dB) between clean and estimated magnitude spectrograms.
    Higher is better. A quick check that works without the .wav files."""
    num = np.sum(clean ** 2, axis=(1, 2))
    den = np.sum((clean - est) ** 2, axis=(1, 2)) + 1e-12
    return 10 * np.log10(num / den + 1e-12)


@torch.no_grad()
def predict_masks(model, noisy, device, batch=16):
    model.eval()
    out = []
    for s in range(0, len(noisy), batch):
        x = torch.from_numpy(noisy[s:s + batch]).to(device)
        out.append(model(x).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def enhance_wav(model, wav, cfg, device):
    """Same STFT as features.py (librosa defaults: centered, zero-padded, periodic Hann)."""
    window = torch.hann_window(cfg["n_fft"], device=device)
    x = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    X = torch.stft(x, cfg["n_fft"], cfg["hop"], window=window, center=True,
                   pad_mode="constant", return_complex=True)
    y = torch.istft(X * model(X.abs()), cfg["n_fft"], cfg["hop"], window=window,
                    center=True, length=x.shape[-1])
    return y.squeeze(0).cpu().numpy()


def summarize(rows, keys, noise_types):
    groups = defaultdict(list)
    for r in rows:
        groups[r["noise_type"]].append(r)
    order = (sorted(groups) + ["ALL"]) if noise_types else ["ALL"]
    for t in order:
        g = rows if t == "ALL" else groups[t]
        parts = []
        for name, before, after, fmt in keys:
            b = np.nanmean([r[before] for r in g])
            a = np.nanmean([r[after] for r in g])
            parts.append(f"{name} {b:{fmt}} -> {a:{fmt}} ({a - b:+{fmt}})")
        print(f"  {t:<10}{len(g):>4} clips   " + "   ".join(parts))


def evaluate(model, cfg, device, noise_types):
    noisy, clean, ids = load_split(cfg, "test")
    est = predict_masks(model, noisy, device) * noisy
    s_noisy, s_est = spec_snr(clean, noisy), spec_snr(clean, est)
    rows = [dict(pair_id=p, noise_type=noise_types.get(p, "unknown"),
                 spec_snr_noisy=float(a), spec_snr_denoised=float(b))
            for p, a, b in zip(ids, s_noisy, s_est)]

    wav_dir = os.path.join(cfg["data_dir"], "test")
    have_wavs = os.path.exists(os.path.join(wav_dir, "noisy", ids[0] + ".wav"))
    if have_wavs:
        import soundfile as sf
        from pesq import pesq
        from pystoi import stoi
        out_dir = os.path.join(cfg["out_dir"], "denoised")
        os.makedirs(out_dir, exist_ok=True)
        for r in rows:
            n, _ = sf.read(os.path.join(wav_dir, "noisy", r["pair_id"] + ".wav"), dtype="float32")
            c, _ = sf.read(os.path.join(wav_dir, "clean", r["pair_id"] + ".wav"), dtype="float32")
            e = enhance_wav(model, n, cfg, device)
            sf.write(os.path.join(out_dir, r["pair_id"] + ".wav"), e, cfg["sr"])
            for tag, sig in [("noisy", n), ("denoised", e)]:
                try:
                    r[f"pesq_{tag}"] = pesq(cfg["sr"], c, sig, "wb")
                except Exception:
                    r[f"pesq_{tag}"] = float("nan")
                r[f"stoi_{tag}"] = stoi(c, sig, cfg["sr"], extended=False)

    with open(os.path.join(cfg["out_dir"], "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== Test set (noisy -> denoised) ===")
    keys = [("Spec-SNR dB", "spec_snr_noisy", "spec_snr_denoised", ".2f")]
    if have_wavs:
        keys += [("PESQ", "pesq_noisy", "pesq_denoised", ".2f"),
                 ("STOI", "stoi_noisy", "stoi_denoised", ".3f")]
    summarize(rows, keys, noise_types)
    print(f"\nPer-clip scores: {os.path.join(cfg['out_dir'], 'results.csv')}")
    if have_wavs:
        print(f"Denoised audio:  {os.path.join(cfg['out_dir'], 'denoised')}  "
              f"(compare with {wav_dir}/noisy and /clean)")
    else:
        print("No test .wav files found, so PESQ/STOI were skipped. "
              "Run `python src/make_pairs.py` to generate them, then rerun.")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    for k in ["feat_dir", "data_dir", "out_dir"]:
        ap.add_argument(f"--{k}", default=CFG[k])
    ap.add_argument("--train_per_type", type=int, default=CFG["train_per_type"])
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    args, _ = ap.parse_known_args()
    cfg = {**CFG, **vars(args)}

    rng = random.Random(cfg["seed"])
    np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    noise_types = read_noise_types(cfg)
    tr_noisy, tr_clean, tr_ids = load_split(cfg, "train")
    va_noisy, va_clean, _ = load_split(cfg, "val")

    idx = pick_train(tr_ids, noise_types, cfg["train_per_type"], rng)
    counts = defaultdict(int)
    for i in idx:
        counts[noise_types.get(tr_ids[i], "unknown")] += 1
    print(f"Train: {len(idx)} clips {dict(counts)}  |  val: {len(va_noisy)}  |  device: {device}")
    with open(os.path.join(cfg["out_dir"], "train_clips.json"), "w") as f:
        json.dump({"train": [tr_ids[i] for i in idx], "cfg": cfg}, f, indent=1)

    train_dl = DataLoader(TensorDataset(torch.from_numpy(tr_noisy[idx]), torch.from_numpy(tr_clean[idx])),
                          batch_size=cfg["batch_size"], shuffle=True)
    val_dl = DataLoader(TensorDataset(torch.from_numpy(va_noisy), torch.from_numpy(va_clean)),
                        batch_size=cfg["batch_size"])
    del tr_noisy, tr_clean   # free the unused training clips

    model = LSTMDenoiser(cfg["n_fft"] // 2 + 1, cfg["hidden"], cfg["layers"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ckpt = os.path.join(cfg["out_dir"], "best_model.pt")
    best, bad = float("inf"), 0
    for ep in range(1, cfg["epochs"] + 1):
        tr = run_epoch(model, train_dl, device, opt)
        va = run_epoch(model, val_dl, device)
        sched.step(va)
        flag = ""
        if va < best:
            best, bad = va, 0
            torch.save(model.state_dict(), ckpt)
            flag = "  * saved"
        else:
            bad += 1
        print(f"Epoch {ep:02d}  train {tr:.5f}  val {va:.5f}{flag}")
        if bad >= cfg["patience"]:
            print("Early stopping.")
            break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    evaluate(model, cfg, device, noise_types)


if __name__ == "__main__":
    main()
