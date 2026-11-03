# Model comparison: DNN vs CNN vs LSTM vs CNN+LSTM

This folder follows the CA's plan from the 10/13 meeting. All three architectures are trained on all 500 training clips with **exactly the same settings**. They are compared on SNR and PESQ (STOI is calculated too), and only then is the best one picked and tuned.

| file | what it does |
|---|---|
| `config.py` | The one shared training recipe (optimizer, LR, batch size, epochs, early stopping, loss). Change it here and it changes for every model. |
| `models.py` | The three networks. All of them read the same noisy spectrogram and output a 0–1 mask. |
| `train.py` | Trains one model, keeps the checkpoint with the best validation loss, and scores it on val and test. |
| `evaluate.py` | Scores a model the same way for everyone: waveform SNR, PESQ, STOI, broken down by noise type and input SNR level. |
| `compare.py` | Collects finished runs into one table and two charts, and picks the winner using validation data only. |
| `tune.py` | Bayesian hyperparameter search with Optuna (the CA's advice, 11/3), scored on validation only. See below. |
| `../../notebooks/optuna_tuning_colab.ipynb` | Runs `tune.py` on a free Colab GPU, end to end. |

## The models (same input and output; only how they read the spectrogram differs)

- **DNN**: Fully connected. Each frame is predicted from itself plus 4 frames on each side, glued into one long vector. It has no memory and no idea that neighbouring frequency bins are related. This is the baseline, as in MathWorks' own *Denoise Speech Using Deep Learning* example. It has about 1.1M parameters.
- **CNN**: A 2D U-Net that treats the spectrogram as an image and slides 3×3 filters over time×frequency patches. It uses the same architecture as Yuki's `src/cnn/model.py` (on `unet-train-eval`) at 16 base channels, so the numbers line up with the team's CNN work. It has about 0.3M parameters; it is smaller because 2D convolutions are expensive on CPU.
- **LSTM**: Reads the spectrogram one frame at a time from left to right and carries a memory. It has 2 layers × 256 units, the same shape as Sophia's `src/train_lstm.py` (on `lstm-model`), and about 1.1M parameters. It is unidirectional, so it could run in real time.
- **CRN (CNN+LSTM)**: Added 11/3, following the CA's "try an LSTM" advice and Neha's 10/13 idea to combine the two. Five conv layers shrink only the *frequency* axis (257 → 9), keeping every time frame. A 2-layer LSTM reads those frames with memory, and transposed convs with U-Net-style skips rebuild the 257-bin mask. It has about 1.7M parameters and is about 40% faster per clip than the U-Net on CPU.

## Shared recipe (`config.py`)

- Input is log1p(|noisy STFT|), standardised with the training-set mean and std. The output is a sigmoid mask × noisy magnitude, and the noisy phase is reused to rebuild audio.
- Optimizer is Adam at LR 1e-3, halved when validation loss stalls for 3 epochs. Batch size is 8, gradients are clipped at 5, and the seed is 42.
- Training runs for up to 60 epochs, with early stopping after 8 epochs without a better validation loss. The best-validation checkpoint is kept.
- The loss is MSE on |X|^0.3 (power-compressed magnitudes, so quiet speech still counts).

## Run it (from the repo root, after `make_pairs.py` + `features.py`)

```bash
pip install torch numpy soundfile librosa pesq pystoi matplotlib
python src/compare/train.py --model dnn
python src/compare/train.py --model cnn       # slowest on CPU (~4 min/epoch); fine on Colab GPU
python src/compare/train.py --model lstm
python src/compare/train.py --model crn
python src/compare/compare.py --runs dnn cnn lstm crn --out comparison_4models
```

Tuning runs change one setting at a time and get their own name:

```bash
python src/compare/train.py --model lstm --run_name lstm_floor0.1 --set mask_floor=0.1
python src/compare/train.py --model lstm --run_name lstm_h384     --arch hidden=384
```

Checkpoints go to `runs/compare/` (git-ignored). Everything worth keeping (configs, loss curves, per-clip scores, summaries and a few denoised test wavs to listen to) goes to `results/model_comparison/<run>/`.

## Bayesian tuning (`tune.py`)

```bash
python src/compare/tune.py --model cnn --n_trials 25        # U-Net (the CA's pick); GPU strongly recommended
python src/compare/tune.py --model cnn --smoke --n_trials 2 # 2-minute plumbing check
```

- **Sampler:** TPE, Optuna's Bayesian method. The first 6 trials are random, then it proposes settings near the best so far. **Search space:** LR (1e-4 to 3e-3, log scale), batch size (4/8/16), mask floor (0 to 0.2), plus U-Net channels (16/24/32) and dropout. The CRN and LSTM get hidden size and dropout instead.
- **Score (maximised):** validation PESQ, minus 1.0 if validation SNR does not improve (the CA's hard rule) and minus 0.5 if validation STOI drops. **The test set is never used while tuning.**
- **Early stopping of weak trials:** every 5 epochs (from epoch 10), a trial is scored with that same score. If it is below the median of earlier trials at that epoch, it is stopped. It is not judged on loss, because a mask floor raises the loss even when it helps PESQ and STOI.
- **Resumable:** the study is a SQLite file (`--storage_dir`, on Drive in Colab). Rerunning the same command continues it. A trial killed mid-run (say, Colab disconnects) is detected through a heartbeat and retried once.
- **Outputs** go to `results/tuning/<study>/`: `trials.csv` (every trial's settings and validation metrics), `best_params.json` (the winning settings plus a `retrain_command`), and `study.db`. Then run the printed command. It retrains the best settings once and scores them on validation **and** test, and its results go next to the other runs in `results/model_comparison/`.
