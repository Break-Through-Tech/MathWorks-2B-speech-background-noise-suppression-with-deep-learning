"""
train.py - train a spectrogram denoiser and score it with PESQ / STOI.

The network only sees magnitudes from dataset/spectrograms. Wavs in dataset/audio
are opened inside the metric functions, to supply noisy phase and the clean reference.

    .venv/bin/python src/cnn/train.py                  # one training run
    .venv/bin/python src/cnn/train.py --smoke          # 32 clips, 2 epochs
    .venv/bin/python src/cnn/train.py --lr 3e-4 --run-name mask_lr3e-4
    .venv/bin/python src/cnn/train.py --eval-checkpoint runs/mask_lr1e-3/best.pt --split test

Change one flag per run. Val picks the checkpoint; test is only used with --eval-checkpoint.
"""
import argparse, csv, json, os, random, sys
from dataclasses import asdict, dataclass
from pathlib import Path

CNN = Path(__file__).resolve().parent
SRC = CNN.parent
ROOT = SRC.parent
# Re-run under the project virtualenv before importing torch.
if __name__ == "__main__":
    _venv_python = ROOT / ".venv" / "bin" / "python"
    if _venv_python.exists() and Path(sys.executable).resolve() != _venv_python.resolve():
        os.execv(str(_venv_python), [str(_venv_python), *sys.argv])

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

for _path in (SRC, CNN):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from features import SR, istft, stft
from model import DenoiseUNet


@dataclass
class TrainConfig:
    run_name: str = "mask_lr1e-3"
    lr: float = 1e-3
    batch_size: int = 8
    base_channels: int = 16
    target: str = "mask"          # "mask" or "map"
    dropout: float = 0.0
    epochs: int = 40
    patience: int = 8
    seed: int = 42
    pesq_every: int = 5
    max_train_samples: int | None = None
    spec_root: str = ""
    audio_root: str = ""
    runs_root: str = ""


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _roots(config):
    spec = Path(config.spec_root) if config.spec_root else ROOT / "dataset" / "spectrograms"
    audio = Path(config.audio_root) if config.audio_root else ROOT / "dataset" / "audio"
    runs = Path(config.runs_root) if config.runs_root else ROOT / "runs"
    return spec, audio, runs


def load_split(split, spec_root, audio_root):
    """Load linear magnitudes and check that every pair_id has noisy and clean wavs."""
    spec_dir = Path(spec_root) / split
    noisy = np.load(spec_dir / "noisy_mag.npy")
    clean = np.load(spec_dir / "clean_mag.npy")
    ids = np.load(spec_dir / "pair_id.npy").astype(str)
    if noisy.shape != clean.shape or noisy.shape[0] != len(ids):
        raise ValueError(f"{split}: noisy {noisy.shape}, clean {clean.shape}, ids {len(ids)}")
    if noisy.ndim != 3 or tuple(noisy.shape[1:]) != (257, 376):
        raise ValueError(f"{split}: expected (N, 257, 376), got {noisy.shape}")
    audio = Path(audio_root) / split
    missing = []
    for pid in ids:
        for kind in ("noisy", "clean"):
            path = audio / kind / f"{pid}.wav"
            if not path.is_file():
                missing.append(str(path))
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{split}: {len(missing)} wavs missing, e.g. {preview}")
    return noisy.astype(np.float32), clean.astype(np.float32), ids


def wav_path(audio_root, split, kind, pair_id):
    return Path(audio_root) / split / kind / f"{pair_id}.wav"


def read_wav(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path}: sample rate {sr}, expected {SR}")
    return wav


def assert_magnitudes_match_wavs(noisy, ids, audio_root, split, n=4):
    """A few stored rows must equal abs(stft(paired noisy wav)). Catches a shuffled import."""
    step = max(1, len(ids) // n)
    worst = 0.0
    checked = []
    for i in list(range(0, len(ids), step))[:n]:
        wav = read_wav(wav_path(audio_root, split, "noisy", ids[i]))
        mag = np.abs(stft(wav)).astype(np.float32)
        if mag.shape != noisy[i].shape:
            raise AssertionError(f"{ids[i]}: stft shape {mag.shape} != stored {noisy[i].shape}")
        diff = float(np.max(np.abs(mag - noisy[i])))
        worst = max(worst, diff)
        checked.append(str(ids[i]))
        if diff > 1e-3:
            raise AssertionError(f"{ids[i]}: spectrogram does not match its wav (max abs diff {diff:.3e})")
    print(f"pairing ok on {split}: {', '.join(checked)} (max abs diff {worst:.2e})")


def reconstruct(magnitude, noisy_wav):
    """Replace the noisy STFT magnitude and invert. Phase comes from the wav, not the network."""
    spectrum = stft(noisy_wav)
    if spectrum.shape != magnitude.shape:
        raise ValueError(f"magnitude {magnitude.shape} does not match stft {spectrum.shape}")
    mixed = magnitude.astype(np.float32) * np.exp(1j * np.angle(spectrum))
    return istft(mixed, length=len(noisy_wav)).astype(np.float32)


def _pesq_stoi(clean, estimate):
    from pesq import pesq as pesq_wb
    from pystoi import stoi
    if len(clean) != len(estimate):
        n = min(len(clean), len(estimate))
        clean, estimate = clean[:n], estimate[:n]
    return float(pesq_wb(SR, clean, estimate, "wb")), float(stoi(clean, estimate, SR, extended=False))


def identity_check(spec_root=None, audio_root=None, split="val", n=4):
    """Stored noisy magnitude + the wav's own phase must score like the noisy wav itself."""
    spec_root = Path(spec_root or ROOT / "dataset" / "spectrograms")
    audio_root = Path(audio_root or ROOT / "dataset" / "audio")
    noisy, _, ids = load_split(split, spec_root, audio_root)
    assert_magnitudes_match_wavs(noisy, ids, audio_root, split, n=n)
    step = max(1, len(ids) // n)
    pesq_gap = stoi_gap = 0.0
    for i in list(range(0, len(ids), step))[:n]:
        pid = str(ids[i])
        noisy_wav = read_wav(wav_path(audio_root, split, "noisy", pid))
        clean_wav = read_wav(wav_path(audio_root, split, "clean", pid))
        roundtrip = reconstruct(noisy[i], noisy_wav)
        base_p, base_s = _pesq_stoi(clean_wav, noisy_wav)
        back_p, back_s = _pesq_stoi(clean_wav, roundtrip)
        pesq_gap = max(pesq_gap, abs(base_p - back_p))
        stoi_gap = max(stoi_gap, abs(base_s - back_s))
        print(f"  {pid}: noisy PESQ {base_p:.3f} vs round-trip {back_p:.3f}, STOI {base_s:.3f} vs {back_s:.3f}")
    if pesq_gap > 0.05 or stoi_gap > 0.01:
        raise AssertionError(f"identity check failed (PESQ gap {pesq_gap:.3f}, STOI gap {stoi_gap:.3f})")
    print(f"identity check passed (max PESQ gap {pesq_gap:.3f}, max STOI gap {stoi_gap:.3f})")
    return {"pesq_gap": pesq_gap, "stoi_gap": stoi_gap}


class SpectrogramDataset(Dataset):
    """Normalized log1p(noisy) as the model input. Linear magnitudes stay available for the loss."""

    def __init__(self, noisy, clean, mean, std):
        self.noisy = noisy
        self.clean = clean
        self.mean = np.float32(mean)
        self.std = np.float32(std)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, index):
        noisy = self.noisy[index]
        clean = self.clean[index]
        x = (np.log1p(noisy) - self.mean) / self.std
        return (
            torch.from_numpy(x.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(noisy),
            torch.from_numpy(clean),
        )


def magnitude_from_output(output, noisy_linear, target):
    """output is (B, 1, F, T). Mask mode multiplies the raw noisy magnitude."""
    pred = output.squeeze(1)
    if target == "mask":
        return pred * noisy_linear
    # log1p of these magnitudes sits near 0–6. Clamp before expm1 so a wild output cannot overflow.
    return torch.expm1(pred.clamp(-1, 12))


def log_mse(pred_mag, clean_mag):
    return nn.functional.mse_loss(torch.log1p(pred_mag.clamp(min=0)), torch.log1p(clean_mag))


def _batch_loss(model, x, noisy, clean, target):
    out = model(x)
    if target == "map":
        return nn.functional.mse_loss(out.squeeze(1), torch.log1p(clean)), magnitude_from_output(out, noisy, target)
    pred = magnitude_from_output(out, noisy, target)
    return log_mse(pred, clean), pred


def run_epoch(model, loader, optimizer, device, target):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for x, noisy, clean in loader:
        x, noisy, clean = x.to(device), noisy.to(device), clean.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss, _ = _batch_loss(model, x, noisy, clean, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{'train' if training else 'val'} loss is {loss.item()}")
        if training:
            loss.backward()
            optimizer.step()
        total += loss.item() * x.shape[0]
        count += x.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def predict_magnitudes(model, noisy, mean, std, batch_size, device):
    """Forward spectrograms only. Returns linear predicted magnitudes, (N, 257, 376)."""
    model.eval()
    pieces = []
    for start in range(0, len(noisy), batch_size):
        chunk = noisy[start:start + batch_size]
        x = ((np.log1p(chunk) - mean) / std).astype(np.float32)
        xt = torch.from_numpy(x).unsqueeze(1).to(device)
        nt = torch.from_numpy(chunk).to(device)
        pred = magnitude_from_output(model(xt), nt, model.target)
        pieces.append(pred.cpu().numpy())
    return np.concatenate(pieces, axis=0)


def _score_pairs(ids, audio_root, split, estimates=None):
    """Score either the noisy wavs (estimates is None) or one enhanced magnitude per id."""
    pesq_scores, stoi_scores, failed = [], [], []
    for i, pid in enumerate(ids):
        pid = str(pid)
        try:
            noisy_wav = read_wav(wav_path(audio_root, split, "noisy", pid))
            clean_wav = read_wav(wav_path(audio_root, split, "clean", pid))
            estimate = noisy_wav if estimates is None else reconstruct(estimates[i], noisy_wav)
            p, s = _pesq_stoi(clean_wav, estimate)
        except Exception as exc:
            failed.append(f"{pid}: {exc}")
            continue
        pesq_scores.append(p)
        stoi_scores.append(s)
    if len(pesq_scores) * 2 < len(ids):
        raise RuntimeError(f"{split}: PESQ/STOI failed on {len(failed)}/{len(ids)} clips, e.g. {failed[:3]}")
    if failed:
        print(f"  skipped {len(failed)} clips, e.g. {failed[0]}")
    return {"n": len(pesq_scores), "pesq": float(np.mean(pesq_scores)), "stoi": float(np.mean(stoi_scores))}


def noisy_baseline(ids, audio_root, split):
    """PESQ/STOI of the unprocessed noisy wav. This is the number the model has to beat."""
    return _score_pairs(ids, audio_root, split, estimates=None)


def score_waveforms(pred_mags, ids, audio_root, split):
    """PESQ wideband and STOI after rebuilding audio with the paired noisy phase."""
    return _score_pairs(ids, audio_root, split, estimates=pred_mags)


def normalization(noisy):
    log_mag = np.log1p(noisy)
    std = float(log_mag.std())
    return float(log_mag.mean()), std if std > 1e-8 else 1.0


def _loader(dataset, batch_size, seed, shuffle):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def _save_checkpoint(path, model, mean, std, config, epoch, val_loss, val_pesq, val_stoi):
    torch.save({
        "model": model.state_dict(),
        "mean": mean,
        "std": std,
        "config": asdict(config),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_pesq": val_pesq,
        "val_stoi": val_stoi,
    }, path)


def load_checkpoint(path, device=None):
    device = device or pick_device()
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = DenoiseUNet(cfg["base_channels"], cfg["dropout"], cfg["target"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def train(config: TrainConfig) -> Path:
    """Train on spectrograms. Returns the run directory (config.json, metrics.csv, best.pt)."""
    if config.target not in ("mask", "map"):
        raise ValueError(f"target must be 'mask' or 'map', got {config.target!r}")
    set_seed(config.seed)
    spec_root, audio_root, runs_root = _roots(config)
    run_dir = runs_root / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_n, train_c, train_ids = load_split("train", spec_root, audio_root)
    val_n, val_c, val_ids = load_split("val", spec_root, audio_root)
    assert_magnitudes_match_wavs(train_n, train_ids, audio_root, "train")
    identity_check(spec_root, audio_root, split="val")
    if config.max_train_samples is not None:
        n = config.max_train_samples
        train_n, train_c, train_ids = train_n[:n], train_c[:n], train_ids[:n]

    mean, std = normalization(train_n)
    device = pick_device()
    model = DenoiseUNet(config.base_channels, config.dropout, config.target).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_loader = _loader(SpectrogramDataset(train_n, train_c, mean, std), config.batch_size, config.seed, True)
    val_loader = _loader(SpectrogramDataset(val_n, val_c, mean, std), config.batch_size, config.seed, False)

    record = asdict(config)
    record.update({"mean": mean, "std": std, "n_train": int(len(train_ids)), "n_val": int(len(val_ids)),
                   "n_params": n_params, "device": str(device)})
    (run_dir / "config.json").write_text(json.dumps(record, indent=2))
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as fh:
        csv.DictWriter(fh, ["epoch", "train_loss", "val_loss", "val_pesq", "val_stoi"]).writeheader()

    print(f"{config.run_name}: {n_params:,} parameters on {device}, train {len(train_ids)}, val {len(val_ids)}")
    print(f"input log1p mean {mean:.4f} std {std:.4f}, target {config.target}")
    baseline = noisy_baseline(val_ids, audio_root, "val")
    record["noisy_pesq"] = baseline["pesq"]
    record["noisy_stoi"] = baseline["stoi"]
    (run_dir / "config.json").write_text(json.dumps(record, indent=2))
    print(f"val noisy baseline  PESQ {baseline['pesq']:.3f}  STOI {baseline['stoi']:.3f}")

    best_pesq, best_loss, stale = None, float("inf"), 0
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, config.target)
        val_loss = run_epoch(model, val_loader, None, device, config.target)
        improved = val_loss < best_loss
        stale = 0 if improved else stale + 1
        if improved:
            best_loss = val_loss

        val_pesq = val_stoi = ""
        score_now = epoch % config.pesq_every == 0 or improved or epoch == config.epochs
        if score_now:
            pred = predict_magnitudes(model, val_n, mean, std, config.batch_size, device)
            scores = score_waveforms(pred, val_ids, audio_root, "val")
            val_pesq, val_stoi = scores["pesq"], scores["stoi"]
            if best_pesq is None or val_pesq > best_pesq:
                best_pesq = val_pesq
                _save_checkpoint(run_dir / "best.pt", model, mean, std, config, epoch, val_loss, val_pesq, val_stoi)
            extra = f"  pesq {val_pesq:.3f} (noisy {baseline['pesq']:.3f})  stoi {val_stoi:.3f} (noisy {baseline['stoi']:.3f})"
        else:
            extra = ""
        print(f"epoch {epoch:02d}/{config.epochs}  train {train_loss:.5f}  val {val_loss:.5f}{extra}")
        with metrics_path.open("a", newline="") as fh:
            csv.DictWriter(fh, ["epoch", "train_loss", "val_loss", "val_pesq", "val_stoi"]).writerow({
                "epoch": epoch, "train_loss": f"{train_loss:.6f}", "val_loss": f"{val_loss:.6f}",
                "val_pesq": val_pesq, "val_stoi": val_stoi,
            })
        if stale >= config.patience:
            print(f"early stop at epoch {epoch}: val loss did not improve for {config.patience} epochs")
            break

    if not (run_dir / "best.pt").exists():
        _save_checkpoint(run_dir / "best.pt", model, mean, std, config, epoch, val_loss, None, None)
    print(f"best val PESQ {best_pesq}  checkpoint {run_dir / 'best.pt'}")
    return run_dir


def evaluate_checkpoint(checkpoint, split="test", batch_size=8):
    """Score one sealed split. Training never calls this."""
    device = pick_device()
    model, ckpt = load_checkpoint(checkpoint, device)
    spec_root = Path(ckpt["config"].get("spec_root") or ROOT / "dataset" / "spectrograms")
    audio_root = Path(ckpt["config"].get("audio_root") or ROOT / "dataset" / "audio")
    noisy, _, ids = load_split(split, spec_root, audio_root)
    pred = predict_magnitudes(model, noisy, ckpt["mean"], ckpt["std"], batch_size, device)
    scores = score_waveforms(pred, ids, audio_root, split)
    baseline = noisy_baseline(ids, audio_root, split)
    scores["noisy_pesq"] = baseline["pesq"]
    scores["noisy_stoi"] = baseline["stoi"]
    print(f"{split} n={scores['n']}  model PESQ {scores['pesq']:.3f}  STOI {scores['stoi']:.3f}")
    print(f"{split} noisy baseline  PESQ {baseline['pesq']:.3f}  STOI {baseline['stoi']:.3f}")
    return scores


def denoise_clip(checkpoint, split, index=0):
    """Return noisy wav, enhanced wav, clean wav, and the three magnitudes for one pair."""
    device = pick_device()
    model, ckpt = load_checkpoint(checkpoint, device)
    spec_root = Path(ckpt["config"]["spec_root"] or ROOT / "dataset" / "spectrograms")
    audio_root = Path(ckpt["config"]["audio_root"] or ROOT / "dataset" / "audio")
    noisy, clean, ids = load_split(split, spec_root, audio_root)
    pid = str(ids[index])
    pred = predict_magnitudes(model, noisy[index:index + 1], ckpt["mean"], ckpt["std"], 1, device)[0]
    noisy_wav = read_wav(wav_path(audio_root, split, "noisy", pid))
    clean_wav = read_wav(wav_path(audio_root, split, "clean", pid))
    enhanced = reconstruct(pred, noisy_wav)
    return {
        "pair_id": pid,
        "sr": SR,
        "noisy_wav": noisy_wav,
        "enhanced_wav": enhanced,
        "clean_wav": clean_wav,
        "noisy_mag": noisy[index],
        "pred_mag": pred,
        "clean_mag": clean[index],
    }


def build_config(args):
    config = TrainConfig(
        run_name=args.run_name,
        lr=args.lr,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        target=args.target,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        pesq_every=args.pesq_every,
        max_train_samples=args.max_train_samples,
    )
    if args.smoke:
        config.run_name = "smoke"
        config.epochs = 2
        config.max_train_samples = 32
        config.pesq_every = 1
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the spectrogram denoiser")
    parser.add_argument("--run-name", default="mask_lr1e-3")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--target", choices=["mask", "map"], default="mask")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pesq-every", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="2 epochs on 32 train clips")
    parser.add_argument("--identity-check", action="store_true", help="check val pairing and exit")
    parser.add_argument("--eval-checkpoint", default=None, help="score this checkpoint instead of training")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args(argv)

    if args.identity_check:
        identity_check()
        return
    if args.eval_checkpoint:
        evaluate_checkpoint(args.eval_checkpoint, split=args.split, batch_size=args.batch_size)
        return
    train(build_config(args))


if __name__ == "__main__":
    main()
