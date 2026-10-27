"""
train.py - train ONE model with the shared recipe in config.py, then score it.

Run from the repo root:
    python src/compare/train.py --model dnn
    python src/compare/train.py --model cnn
    python src/compare/train.py --model lstm

Tuning runs (after the comparison) change one setting and get their own name:
    python src/compare/train.py --model lstm --run_name lstm_floor0.1 --set mask_floor=0.1
    python src/compare/train.py --model lstm --run_name lstm_h384 --arch hidden=384

Quick smoke test (tiny, ~1 min):  add  --smoke

Outputs
    runs/compare/<run>/best.pt                    checkpoint (lowest validation loss)
    results/model_comparison/<run>/
        config.json          every setting used + parameter count + training time
        history.csv          train / val loss per epoch
        {val,test}_summary.json, {val,test}_per_clip.csv
        denoised_wavs/       first few test clips, to listen to
"""
import argparse, ast, csv, json, os, random, sys, time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import SHARED                       # noqa: E402
from models import build_model, count_params    # noqa: E402
from evaluate import evaluate, print_summary    # noqa: E402

RESULTS_DIR = "results/model_comparison"


def parse_kv(items):
    out = {}
    for item in items or []:
        k, v = item.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load(cfg, split):
    d = np.load(os.path.join(cfg["feat_dir"], f"{split}.npz"))
    return d["noisy_mag"], d["clean_mag"]


def make_loss(cfg):
    p = cfg["compress_power"]

    def loss_fn(est, clean):
        return torch.mean(((est + 1e-8) ** p - (clean + 1e-8) ** p) ** 2)
    return loss_fn


def run_epoch(model, loader, loss_fn, device, opt=None, clip=5.0):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            loss = loss_fn(model(noisy) * noisy, clean)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
            total += loss.item() * len(noisy)
            n += len(noisy)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dnn", "cnn", "lstm"])
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--set", nargs="*", help="override shared settings, e.g. lr=3e-4 mask_floor=0.1")
    ap.add_argument("--arch", nargs="*", help="architecture settings, e.g. hidden=384")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    cfg = {**SHARED, **parse_kv(args.set)}
    arch = parse_kv(args.arch)
    run = args.run_name or args.model
    if args.smoke:
        cfg.update(num_train_clips=24, max_epochs=2)
        run = f"smoke_{run}"
    if args.threads:
        torch.set_num_threads(args.threads)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(cfg["runs_dir"], run)
    res_dir = os.path.join(RESULTS_DIR, run)
    os.makedirs(run_dir, exist_ok=True); os.makedirs(res_dir, exist_ok=True)

    tr_noisy, tr_clean = load(cfg, "train")
    tr_noisy, tr_clean = tr_noisy[:cfg["num_train_clips"]], tr_clean[:cfg["num_train_clips"]]
    va_noisy, va_clean = load(cfg, "val")

    model = build_model(args.model, mask_floor=cfg["mask_floor"], **arch).to(device)
    logmag = np.log1p(tr_noisy)
    model.set_norm(logmag.mean(), logmag.std())
    del logmag
    n_params = count_params(model)

    g = torch.Generator().manual_seed(cfg["seed"])
    train_dl = DataLoader(TensorDataset(torch.from_numpy(tr_noisy), torch.from_numpy(tr_clean)),
                          batch_size=cfg["batch_size"], shuffle=True, generator=g)
    val_dl = DataLoader(TensorDataset(torch.from_numpy(va_noisy), torch.from_numpy(va_clean)),
                        batch_size=cfg["batch_size"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=cfg["lr_decay_factor"], patience=cfg["lr_decay_patience"])
    loss_fn = make_loss(cfg)

    print(f"[{run}] model={args.model} params={n_params:,} train={len(tr_noisy)} val={len(va_noisy)} "
          f"device={device} threads={torch.get_num_threads()}", flush=True)

    ckpt = os.path.join(run_dir, "best.pt")
    last = os.path.join(run_dir, "last.pt")      # lets a long (CPU) run resume where it stopped
    best, best_epoch, bad, history, start, prev_minutes = float("inf"), 0, 0, [], 1, 0.0
    if os.path.exists(last) and not args.smoke:
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng"]); g.set_state(st["gen"])
        best, best_epoch, bad, history = st["best"], st["best_epoch"], st["bad"], st["history"]
        start, prev_minutes = st["epoch"] + 1, st["minutes"]
        print(f"[{run}] resuming after epoch {st['epoch']}", flush=True)
    t0 = time.time()
    stopped = bad >= cfg["early_stop_patience"]
    for ep in range(start, cfg["max_epochs"] + 1):
        if stopped:
            break
        te = time.time()
        tr = run_epoch(model, train_dl, loss_fn, device, opt, cfg["grad_clip"])
        va = run_epoch(model, val_dl, loss_fn, device)
        sched.step(va)
        improved = va < best
        if improved:
            best, best_epoch, bad = va, ep, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        history.append(dict(epoch=ep, train_loss=tr, val_loss=va, lr=opt.param_groups[0]["lr"],
                             seconds=round(time.time() - te, 1)))
        print(f"[{run}] epoch {ep:02d}  train {tr:.5f}  val {va:.5f}  lr {opt.param_groups[0]['lr']:.1e}  "
              f"{time.time() - te:.0f}s{'  * best' if improved else ''}", flush=True)
        stopped = bad >= cfg["early_stop_patience"]
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                        rng=torch.get_rng_state(), gen=g.get_state(), best=best, best_epoch=best_epoch,
                        bad=bad, history=history, epoch=ep,
                        minutes=prev_minutes + (time.time() - t0) / 60), last)
        if stopped:
            print(f"[{run}] early stop: no val improvement for {bad} epochs", flush=True)
    train_minutes = prev_minutes + (time.time() - t0) / 60

    with open(os.path.join(res_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader(); w.writerows(history)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    summaries = {}
    for split in ("val", "test"):
        _, s = evaluate(model, split, cfg, device, out_dir=res_dir, save_wavs=5 if split == "test" else 0)
        summaries[split] = s
        with open(os.path.join(res_dir, f"{split}_summary.json"), "w") as f:
            json.dump(s, f, indent=1)
        print_summary(run, split, s)

    with open(os.path.join(res_dir, "config.json"), "w") as f:
        json.dump(dict(run=run, model=args.model, arch=arch, n_params=n_params,
                       best_epoch=best_epoch, epochs_run=len(history), best_val_loss=best,
                       train_minutes=round(train_minutes, 1), shared=cfg), f, indent=1)
    print(f"[{run}] done: best epoch {best_epoch}, {train_minutes:.1f} min. Results in {res_dir}", flush=True)


if __name__ == "__main__":
    main()
