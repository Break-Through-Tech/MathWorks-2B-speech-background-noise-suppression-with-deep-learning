"""
compare.py - collect every finished run into one table + charts.

    python src/compare/compare.py                      # the 3-model comparison
    python src/compare/compare.py --runs lstm lstm_floor0.1 ...   --out tuning

How the winner is picked (validation set only, so the test set stays a fair final check):
  1. the CA's hard rule: average SNR must go UP vs the noisy input
  2. among models that pass, highest PESQ on validation
Test numbers are reported next to it but never used for choosing.

Writes results/model_comparison/<out>.md, <out>.csv and <out>_*.png
"""
import argparse, csv, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = "results/model_comparison"
COLORS = {"dnn": "#8a8a8a", "cnn": "#2a6fdb", "lstm": "#d9822b"}


def load_run(run):
    d = os.path.join(RESULTS, run)
    out = {"run": run}
    with open(os.path.join(d, "config.json")) as f:
        out["config"] = json.load(f)
    for split in ("val", "test"):
        with open(os.path.join(d, f"{split}_summary.json")) as f:
            out[split] = json.load(f)
    with open(os.path.join(d, "history.csv")) as f:
        out["history"] = list(csv.DictReader(f))
    return out


def pick_winner(runs):
    passing = [r for r in runs if r["val"]["overall"]["snr_gain"] > 0]
    pool = passing or runs
    return max(pool, key=lambda r: r["val"]["overall"]["pesq_out"])["run"]


def table_rows(runs, winner):
    rows = []
    for r in runs:
        c, v, t = r["config"], r["val"]["overall"], r["test"]["overall"]
        rows.append(dict(
            run=r["run"], model=c["model"], params=c["n_params"], best_epoch=c["best_epoch"],
            train_min=c["train_minutes"],
            val_snr_gain=round(v["snr_gain"], 2), val_pesq=round(v["pesq_out"], 3),
            test_snr_in=round(t["snr_in"], 2), test_snr_out=round(t["snr_out"], 2),
            test_snr_gain=round(t["snr_gain"], 2), test_pct_snr_worse=round(t["pct_snr_worse"]),
            test_pesq_in=round(t["pesq_in"], 3), test_pesq_out=round(t["pesq_out"], 3),
            test_pesq_gain=round(t["pesq_gain"], 3),
            test_stoi_in=round(t["stoi_in"], 3), test_stoi_out=round(t["stoi_out"], 3),
            selected="yes" if r["run"] == winner else "",
        ))
    return rows


def color(r):
    return COLORS.get(r["run"], None) or COLORS.get(r["config"]["model"], None)


def plot_by_level(runs, out_prefix, split="test"):
    levels = list(runs[0][split]["by_snr_level"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.8 / len(runs)
    for ax, key, title in ((axes[0], "snr_gain", "SNR change (dB)"), (axes[1], "pesq_gain", "PESQ change")):
        for i, r in enumerate(runs):
            vals = [r[split]["by_snr_level"][lv][key] for lv in levels]
            xs = [j + (i - (len(runs) - 1) / 2) * width for j in range(len(levels))]
            ax.bar(xs, vals, width, label=r["run"], color=color(r) if len(runs) <= 3 else None)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"{lv} dB" for lv in levels])
        ax.set_xlabel("input SNR of the noisy clip")
        ax.set_title(f"{title} vs noisy input ({split} set)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path = f"{out_prefix}_by_snr_level.png"
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_curves(runs, out_prefix):
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in runs:
        ep = [int(h["epoch"]) for h in r["history"]]
        ax.plot(ep, [float(h["val_loss"]) for h in r["history"]], label=f"{r['run']} val",
                color=color(r) if len(runs) <= 3 else None)
        ax.plot(ep, [float(h["train_loss"]) for h in r["history"]], ls="--", alpha=0.5,
                color=ax.lines[-1].get_color())
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (solid = val, dashed = train)")
    ax.set_title("Training curves")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = f"{out_prefix}_training_curves.png"
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def write_markdown(runs, rows, winner, out_prefix, title):
    L = [f"# {title}", ""]
    L.append("All runs: same 500 training clips, same 50 val / 50 test clips, same recipe "
             "(`src/compare/config.py`). Winner = SNR must improve on val, then highest val PESQ. "
             "Test numbers are the final, untouched check.\n")
    L.append("| run | params | best epoch | train min | test SNR in→out (dB) | ΔSNR | clips worse | "
             "test PESQ in→out | ΔPESQ | test STOI in→out | picked |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['run']} | {r['params']:,} | {r['best_epoch']} | {r['train_min']} | "
                 f"{r['test_snr_in']} → {r['test_snr_out']} | {r['test_snr_gain']:+} | {r['test_pct_snr_worse']}% | "
                 f"{r['test_pesq_in']} → {r['test_pesq_out']} | {r['test_pesq_gain']:+} | "
                 f"{r['test_stoi_in']} → {r['test_stoi_out']} | {'✅' if r['selected'] else ''} |")
    L.append("\n## Test set by input SNR level (ΔSNR dB / ΔPESQ)\n")
    levels = list(runs[0]["test"]["by_snr_level"].keys())
    L.append("| run | " + " | ".join(f"{lv} dB" for lv in levels) + " |")
    L.append("|---|" + "---|" * len(levels))
    for r in runs:
        cells = [f"{r['test']['by_snr_level'][lv]['snr_gain']:+.2f} / {r['test']['by_snr_level'][lv]['pesq_gain']:+.2f}"
                 for lv in levels]
        L.append(f"| {r['run']} | " + " | ".join(cells) + " |")
    L.append("\n## Test set by noise type (ΔSNR dB / ΔPESQ)\n")
    types = list(runs[0]["test"]["by_noise_type"].keys())
    L.append("| run | " + " | ".join(types) + " |")
    L.append("|---|" + "---|" * len(types))
    for r in runs:
        cells = [f"{r['test']['by_noise_type'][t]['snr_gain']:+.2f} / {r['test']['by_noise_type'][t]['pesq_gain']:+.2f}"
                 for t in types]
        L.append(f"| {r['run']} | " + " | ".join(cells) + " |")
    base = os.path.basename(out_prefix)
    L.append(f"\n![by SNR level]({base}_by_snr_level.png)\n\n![training curves]({base}_training_curves.png)\n")
    with open(f"{out_prefix}.md", "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["dnn", "cnn", "lstm"])
    ap.add_argument("--out", default="comparison")
    ap.add_argument("--title", default="DNN vs CNN vs LSTM - all 500 training clips")
    args = ap.parse_args()

    runs = [load_run(r) for r in args.runs]
    winner = pick_winner(runs)
    rows = table_rows(runs, winner)
    prefix = os.path.join(RESULTS, args.out)
    with open(f"{prefix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    plot_by_level(runs, prefix)
    plot_curves(runs, prefix)
    write_markdown(runs, rows, winner, prefix, args.title)
    for r in rows:
        print(r)
    print(f"\nPicked (val): {winner}\nWrote {prefix}.md / .csv / *.png")


if __name__ == "__main__":
    main()
