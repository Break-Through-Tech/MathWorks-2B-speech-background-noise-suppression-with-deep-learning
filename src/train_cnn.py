"""
CNN speech denoiser for the MathWorks 2B pipeline (cnn-model branch).

Same setup as src/train_lstm.py so the two models can be compared directly:
  * same features   data/processed/features/{train,val,test}.npz   [N, 257, 376] magnitudes
  * same 100 training clips, read from results/lstm_v1_train_clips.json
  * same loss, optimizer, early stopping, and test-set evaluation (imported from train_lstm.py)
Only the model is different.

How the CNN works: the noisy spectrogram is treated like a 1-channel image (257 freq x 376 frames).
Small 3x3 filters slide over it and learn local time-frequency patterns (speech harmonics vs.
fan hum vs. typing clicks). Dilated layers widen the view along time without adding parameters.
The last layer outputs a mask in [0, 1] for every bin: denoised = mask * noisy.

Run from the repo root (after make_pairs.py and features.py):
  python src/train_cnn.py
  python src/train_cnn.py --channels 64      # bigger model
"""
import os, json, random, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# reuse the LSTM script's data loading, loss, training loop and evaluation unchanged
from train_lstm import load_split, read_noise_types, run_epoch, evaluate

CFG = dict(
    feat_dir="data/processed/features",
    data_dir="data/processed",
    out_dir="runs/cnn_v1",
    train_clips="results/lstm_v1_train_clips.json",   # the LSTM's 100 clips -> fair comparison
    sr=16000,
    n_fft=512,
    hop=128,
    channels=32,
    batch_size=8,
    lr=1e-3,
    epochs=80,
    patience=10,         # stop if validation loss hasn't improved for this many epochs
    seed=42,
)


# ---------------------------------------------------------------- model
def conv_block(c_in, c_out, dilation=1):
    # padding = dilation keeps the output the same size as the input (3x3 kernel)
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=(1, dilation), dilation=(1, dilation)),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    )


class CNNDenoiser(nn.Module):
    """Fully convolutional mask estimator. Time dilations 1-2-4-8 give each output bin about
    33 frames (~0.26 s) of context. Uses past and future frames, so it is not streaming."""

    def __init__(self, channels=32):
        super().__init__()
        c = channels
        self.net = nn.Sequential(
            conv_block(1, c),
            conv_block(c, c, dilation=2),
            conv_block(c, c, dilation=4),
            conv_block(c, c, dilation=8),
            conv_block(c, c),
            nn.Conv2d(c, 1, kernel_size=1),
        )

    def forward(self, mag):                      # mag: (B, 257, frames)
        x = torch.log1p(mag).unsqueeze(1)        # (B, 1, 257, frames)  "1-channel image"
        return torch.sigmoid(self.net(x)).squeeze(1)   # mask (B, 257, frames)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    for k in ["feat_dir", "data_dir", "out_dir", "train_clips"]:
        ap.add_argument(f"--{k}", default=CFG[k])
    ap.add_argument("--channels", type=int, default=CFG["channels"])
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    args, _ = ap.parse_known_args()
    cfg = {**CFG, **vars(args)}

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    noise_types = read_noise_types(cfg)
    tr_noisy, tr_clean, tr_ids = load_split(cfg, "train")
    va_noisy, va_clean, _ = load_split(cfg, "val")

    with open(cfg["train_clips"]) as f:
        wanted = json.load(f)["train"]
    pos = {pid: i for i, pid in enumerate(tr_ids)}
    missing = [p for p in wanted if p not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} clips from {cfg['train_clips']} are not in train.npz "
                         f"(e.g. {missing[:3]}). Rerun make_pairs.py and features.py with defaults.")
    idx = sorted(pos[p] for p in wanted)

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

    model = CNNDenoiser(cfg["channels"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ckpt = os.path.join(cfg["out_dir"], "best_model.pt")
    history = []
    best, bad = float("inf"), 0
    for ep in range(1, cfg["epochs"] + 1):
        tr = run_epoch(model, train_dl, device, opt)
        va = run_epoch(model, val_dl, device)
        sched.step(va)
        history.append(dict(epoch=ep, train=tr, val=va))
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

    with open(os.path.join(cfg["out_dir"], "loss_history.json"), "w") as f:
        json.dump(history, f, indent=1)   # for the loss-curve plot in the team update

    model.load_state_dict(torch.load(ckpt, map_location=device))
    evaluate(model, cfg, device, noise_types)


if __name__ == "__main__":
    main()
