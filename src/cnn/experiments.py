"""
experiments.py - train the comparison runs, score test, and write one results table.

    .venv/bin/python src/cnn/experiments.py
    .venv/bin/python src/cnn/experiments.py --only mask_lr1e-3
    .venv/bin/python src/cnn/experiments.py --report
    .venv/bin/python src/cnn/experiments.py --force

Validation PESQ picks the selected row. Test scores are recorded in the table
and are not used to choose a later run.
"""
import argparse, csv, json, os, sys
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

import torch

from train import TrainConfig, evaluate_checkpoint, train


COLUMNS = [
    "run", "lr", "channels", "target", "dropout",
    "best_epoch", "train_loss", "val_loss", "loss_gap",
    "val_pesq", "val_stoi", "test_pesq", "test_stoi", "selected",
]

# Each entry changes one knob from the first row. Seed, epochs, and patience stay on TrainConfig defaults.
SUITE = [
    TrainConfig(run_name="mask_lr1e-3", lr=1e-3, base_channels=16, target="mask", dropout=0.0),
    TrainConfig(run_name="mask_lr3e-4", lr=3e-4, base_channels=16, target="mask", dropout=0.0),
    TrainConfig(run_name="mask_ch32", lr=1e-3, base_channels=32, target="mask", dropout=0.0),
    TrainConfig(run_name="map_lr1e-3", lr=1e-3, base_channels=16, target="map", dropout=0.0),
    TrainConfig(run_name="mask_drop0.2", lr=1e-3, base_channels=16, target="mask", dropout=0.2),
]


def runs_root():
    return ROOT / "runs"


def run_folder(config, root=None):
    return Path(root or runs_root()) / config.run_name


def _load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _losses_at_epoch(metrics_path, epoch):
    if not metrics_path.is_file():
        return None, None
    with metrics_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["epoch"]) == int(epoch):
                return float(row["train_loss"]), float(row["val_loss"])
    return None, None


def _blank(run_name="noisy"):
    row = {column: "" for column in COLUMNS}
    row["run"] = run_name
    row["selected"] = ""
    return row


def _fmt(column, value):
    if value is None or value == "":
        return ""
    if column in ("train_loss", "val_loss", "loss_gap"):
        return f"{float(value):.5f}"
    if column in ("val_pesq", "val_stoi", "test_pesq", "test_stoi"):
        return f"{float(value):.3f}"
    if column == "lr":
        return f"{float(value):.4g}"
    if column == "dropout":
        return f"{float(value):.1f}" if float(value) else "0"
    if column in ("channels", "best_epoch"):
        return str(int(value))
    return str(value)


def run_record(config, root=None):
    """One table row from a finished run. Returns None if best.pt or test.json is missing."""
    folder = run_folder(config, root)
    checkpoint_path = folder / "best.pt"
    test_path = folder / "test.json"
    if not checkpoint_path.is_file() or not test_path.is_file():
        return None
    checkpoint = _load_torch(checkpoint_path)
    saved = checkpoint["config"]
    test = json.loads(test_path.read_text())
    config_path = folder / "config.json"
    record = json.loads(config_path.read_text()) if config_path.is_file() else {}
    epoch = checkpoint.get("epoch")
    train_loss, val_loss = _losses_at_epoch(folder / "metrics.csv", epoch) if epoch is not None else (None, None)
    if val_loss is None and checkpoint.get("val_loss") is not None:
        val_loss = float(checkpoint["val_loss"])
    gap = None if train_loss is None or val_loss is None else val_loss - train_loss
    row = _blank(config.run_name)
    row.update({
        "lr": saved["lr"],
        "channels": saved["base_channels"],
        "target": saved["target"],
        "dropout": saved["dropout"],
        "best_epoch": epoch if epoch is not None else "",
        "train_loss": train_loss if train_loss is not None else "",
        "val_loss": val_loss if val_loss is not None else "",
        "loss_gap": gap if gap is not None else "",
        "val_pesq": checkpoint.get("val_pesq") if checkpoint.get("val_pesq") is not None else "",
        "val_stoi": checkpoint.get("val_stoi") if checkpoint.get("val_stoi") is not None else "",
        "test_pesq": test["pesq"],
        "test_stoi": test["stoi"],
        "selected": "no",
        "_noisy_val_pesq": record.get("noisy_pesq", ""),
        "_noisy_val_stoi": record.get("noisy_stoi", ""),
        "_noisy_test_pesq": test.get("noisy_pesq", ""),
        "_noisy_test_stoi": test.get("noisy_stoi", ""),
    })
    return row


def _baseline_row(rows):
    baseline = _blank("noisy")
    for row in rows:
        if row.get("_noisy_val_pesq") != "":
            baseline["val_pesq"] = row["_noisy_val_pesq"]
            baseline["val_stoi"] = row["_noisy_val_stoi"]
        if row.get("_noisy_test_pesq") != "":
            baseline["test_pesq"] = row["_noisy_test_pesq"]
            baseline["test_stoi"] = row["_noisy_test_stoi"]
        if baseline["val_pesq"] != "" and baseline["test_pesq"] != "":
            break
    return baseline


def _mark_selected(rows):
    ranked = [row for row in rows if row["val_pesq"] != ""]
    if not ranked:
        return
    winner = max(ranked, key=lambda row: (float(row["val_pesq"]), float(row["val_stoi"] or 0)))
    for row in rows:
        row["selected"] = "yes" if row is winner else "no"


def collect_rows(configs, root=None):
    finished = []
    for config in configs:
        row = run_record(config, root)
        if row is not None:
            finished.append(row)
    _mark_selected(finished)
    table = [_baseline_row(finished), *finished] if finished else []
    for row in table:
        row.pop("_noisy_val_pesq", None)
        row.pop("_noisy_val_stoi", None)
        row.pop("_noisy_test_pesq", None)
        row.pop("_noisy_test_stoi", None)
    return table


def write_summary(configs, root=None):
    root = Path(root or runs_root())
    root.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(configs, root)
    path = root / "summary.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(column, row[column]) for column in COLUMNS})
    temporary.replace(path)
    return path, rows


def print_summary(path=None):
    """Print runs/summary.csv. Used by the notebook after the suite has been run."""
    path = Path(path or runs_root() / "summary.csv")
    if not path.is_file():
        print("No summary yet. Run: .venv/bin/python src/cnn/experiments.py")
        return
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"{path} is empty")
        return
    widths = {column: len(column) for column in COLUMNS}
    rendered = []
    for row in rows:
        cells = {column: row.get(column, "") for column in COLUMNS}
        rendered.append(cells)
        for column, text in cells.items():
            widths[column] = max(widths[column], len(text))
    header = "  ".join(column.ljust(widths[column]) for column in COLUMNS)
    print(header)
    print("  ".join("-" * widths[column] for column in COLUMNS))
    for cells in rendered:
        print("  ".join(cells[column].ljust(widths[column]) for column in COLUMNS))
    print(path)
    print("selected is the highest validation PESQ. Test scores are recorded, not used to pick the row.")
    print("A large loss_gap while validation PESQ falls is overfitting. A small gap with validation PESQ near the noisy row is underfitting.")


def _score_test(config, root):
    folder = run_folder(config, root)
    scores = evaluate_checkpoint(folder / "best.pt", split="test", batch_size=config.batch_size)
    (folder / "test.json").write_text(json.dumps(scores, indent=2))


def execute(configs, force=False, report=False, root=None):
    root = Path(root or runs_root())
    for config in configs:
        folder = run_folder(config, root)
        checkpoint = folder / "best.pt"
        if report:
            if not checkpoint.is_file():
                print(f"skip {config.run_name}: no checkpoint")
                continue
        elif force or not checkpoint.is_file():
            train(config)
        else:
            print(f"skip train {config.run_name}: {checkpoint}")
        if not checkpoint.is_file():
            continue
        test_path = folder / "test.json"
        if force or not test_path.is_file():
            _score_test(config, root)
        else:
            print(f"skip test {config.run_name}: {test_path}")
        summary, _ = write_summary(SUITE, root)
        print_summary(summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train comparison runs and write runs/summary.csv")
    parser.add_argument("--only", default=None, help="run name to train and score, e.g. mask_lr1e-3")
    parser.add_argument("--report", action="store_true", help="do not train; score any checkpoint that has no test.json")
    parser.add_argument("--force", action="store_true", help="retrain even when best.pt already exists")
    args = parser.parse_args(argv)

    configs = SUITE
    if args.only:
        configs = [config for config in SUITE if config.run_name == args.only]
        if not configs:
            names = ", ".join(config.run_name for config in SUITE)
            parser.error(f"unknown run {args.only!r}. Choose from: {names}")
    if args.force and args.report:
        parser.error("use either --force or --report")
    execute(configs, force=args.force, report=args.report)


if __name__ == "__main__":
    main()
