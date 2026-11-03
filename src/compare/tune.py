"""
tune.py - Bayesian hyperparameter search with Optuna (CA's advice, 11/3: "start with Bayesian").

    python src/compare/tune.py --model cnn --n_trials 25                 # U-Net (CA's pick)
    python src/compare/tune.py --model crn --n_trials 25                 # CNN+LSTM
    python src/compare/tune.py --model cnn --n_trials 2 --smoke          # 2-minute check

How it works
  * Sampler: TPE (Optuna's Bayesian method). The first `--startup` trials are random to
    explore, then it proposes settings near the ones that scored best so far.
  * Pruner: every 5 epochs the trial is scored on the validation set with the SAME score
    used to rank trials (below); a trial clearly behind the median of earlier trials at that
    epoch is stopped early. (Not judged on loss: a mask floor raises the loss even when it
    improves PESQ/STOI, so loss-based pruning would unfairly kill those trials.)
  * Every trial trains on the 500 training clips and is scored on the VALIDATION set only.
    The test set is never touched while tuning, so the final test score stays honest.
  * Score to maximise = validation PESQ, with the CA's rules as penalties:
        SNR must go up (hard rule)  -> -1.0 if val SNR does not improve
        STOI should not drop        -> -0.5 if val STOI goes down
  * The study is saved in a SQLite file, so a run that stops (e.g. Colab disconnects)
    continues where it left off when you run the same command again.

Outputs (results/tuning/<study>/)
  trials.csv        every trial: settings, val PESQ / SNR / STOI, state
  best_params.json  the winning settings + the command to retrain them on all data
  study.db          the Optuna study (resume / inspect)
After the search:  python src/compare/train.py --model cnn --run_name cnn_optuna_best --set ... --arch ...
(printed for you) trains the best settings once and scores them on val AND test.
"""
import argparse, json, os, shutil, sys, warnings

import optuna
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
from optuna.storages import RDBStorage
try:                                    # newer Optuna name (>= 4.9); older versions use the old one
    from optuna.storages import RetryHeartbeatStaleTrialCallback as RetryCallback
except ImportError:
    from optuna.storages import RetryFailedTrialCallback as RetryCallback
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import SHARED   # noqa: E402
from train import fit       # noqa: E402
from evaluate import evaluate  # noqa: E402

TUNING_DIR = "results/tuning"


def make_storage(db_path):
    """SQLite study with a heartbeat, so a trial killed mid-run is retried once on restart.
    Works with old and new Optuna (the argument was renamed in 4.9)."""
    import inspect
    kw = dict(heartbeat_interval=60, grace_period=180)
    arg = ("heartbeat_stale_trial_callback"
           if "heartbeat_stale_trial_callback" in inspect.signature(RDBStorage).parameters
           else "failed_trial_callback")
    kw[arg] = RetryCallback(max_retry=1)
    return RDBStorage(f"sqlite:///{os.path.abspath(db_path)}", **kw)


def search_space(trial, model):
    """Settings Optuna may change. Everything else stays at config.SHARED."""
    shared = dict(
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [4, 8, 16]),
        mask_floor=trial.suggest_float("mask_floor", 0.0, 0.2),
    )
    if model == "cnn":
        arch = dict(base_channels=trial.suggest_categorical("base_channels", [16, 24, 32]),
                    dropout=trial.suggest_float("dropout", 0.0, 0.3))
    elif model == "crn":
        arch = dict(hidden=trial.suggest_categorical("hidden", [128, 256, 384]),
                    dropout=trial.suggest_float("dropout", 0.0, 0.4))
    elif model == "lstm":
        arch = dict(hidden=trial.suggest_categorical("hidden", [256, 384, 512]),
                    layers=trial.suggest_int("layers", 2, 3),
                    dropout=trial.suggest_float("dropout", 0.0, 0.4))
    else:
        raise ValueError(f"no search space for {model}")
    return shared, arch


def score(val):
    o = val["overall"]
    s = o["pesq_out"]
    if o["snr_gain"] <= 0:
        s -= 1.0
    if o["stoi_gain"] < 0:
        s -= 0.5
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn", choices=["cnn", "crn", "lstm"])
    ap.add_argument("--study", default=None, help="name (default: <model>_optuna)")
    ap.add_argument("--n_trials", type=int, default=25, help="trials to run in THIS call")
    ap.add_argument("--max_epochs", type=int, default=40, help="per trial (early stopping still applies)")
    ap.add_argument("--startup", type=int, default=6, help="random trials before TPE kicks in")
    ap.add_argument("--prune_every", type=int, default=5, help="score on val every N epochs")
    ap.add_argument("--storage_dir", default=None, help="where study.db lives (e.g. a Drive folder on Colab)")
    ap.add_argument("--smoke", action="store_true", help="tiny data, 2 epochs - checks the plumbing only")
    args = ap.parse_args()

    study_name = args.study or f"{args.model}_optuna" + ("_smoke" if args.smoke else "")
    out_dir = os.path.join(TUNING_DIR, study_name)
    os.makedirs(out_dir, exist_ok=True)
    db_dir = args.storage_dir or out_dir
    os.makedirs(db_dir, exist_ok=True)
    # heartbeat: if a run dies mid-trial (e.g. Colab disconnects), that trial is marked failed
    # on the next start and re-queued once with the same settings
    storage = make_storage(os.path.join(db_dir, "study.db"))

    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SHARED["seed"], n_startup_trials=args.startup),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=args.startup, n_warmup_steps=10,
                                           interval_steps=5),
    )

    def objective(trial):
        shared, arch = search_space(trial, args.model)
        cfg = {**SHARED, **shared, "max_epochs": args.max_epochs,
               "runs_dir": os.path.join("runs", "tuning", study_name)}
        if args.smoke:
            cfg.update(num_train_clips=24, max_epochs=2)
            args.prune_every = 1
        run = f"trial_{trial.number:03d}"

        def report(ep, val_loss, model, device):
            if ep % args.prune_every:
                return
            _, val = evaluate(model, "val", cfg, device)      # current weights, val set only
            trial.report(score(val), ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            res = fit(args.model, cfg, arch, run, os.path.join(out_dir, "trials", run),
                      splits=("val",), save_wavs=0, resume=False, on_epoch=report)
        finally:
            shutil.rmtree(os.path.join(cfg["runs_dir"], run), ignore_errors=True)   # free disk
        o = res["summaries"]["val"]["overall"]
        for k in ("pesq_out", "snr_gain", "stoi_out", "stoi_gain", "pct_snr_worse"):
            trial.set_user_attr(f"val_{k}", round(o[k], 4))
        trial.set_user_attr("best_epoch", res["best_epoch"])
        trial.set_user_attr("train_minutes", res["train_minutes"])
        return score(res["summaries"]["val"])

    done = len([t for t in study.trials if t.state.is_finished()])
    print(f"study {study_name}: {done} trials already finished, running {args.n_trials} more "
          f"(device: {'cuda' if torch.cuda.is_available() else 'cpu'})", flush=True)
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    df = study.trials_dataframe()
    df.to_csv(os.path.join(out_dir, "trials.csv"), index=False)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        print("no completed trials yet")
        return
    best = study.best_trial
    shared_keys = {"lr", "batch_size", "mask_floor"}
    sets = " ".join(f"{k}={v}" for k, v in best.params.items() if k in shared_keys)
    archs = " ".join(f"{k}={v}" for k, v in best.params.items() if k not in shared_keys)
    cmd = (f"python src/compare/train.py --model {args.model} --run_name {args.model}_optuna_best "
           f"--set {sets}" + (f" --arch {archs}" if archs else ""))
    summary = dict(study=study_name, model=args.model, best_trial=best.number, score=best.value,
                   params=best.params, val_metrics=best.user_attrs,
                   n_complete=len(complete),
                   n_pruned=len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                   retrain_command=cmd)
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"\nNext: retrain the best settings and score on val + test:\n  {cmd}")


if __name__ == "__main__":
    main()
