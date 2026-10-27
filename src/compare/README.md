# Model comparison: DNN vs CNN vs LSTM

This folder follows the CA's plan from the 10/13 meeting. All three architectures are trained on all 500 training clips with **exactly the same settings**. They are compared on SNR and PESQ (STOI is calculated too), and only then is the best one picked and tuned.

| file | what it does |
|---|---|
| `config.py` | The one shared training recipe (optimizer, LR, batch size, epochs, early stopping, loss). Change it here and it changes for every model. |
| `models.py` | The three networks. All of them read the same noisy spectrogram and output a 0–1 mask. |
| `train.py` | Trains one model, keeps the checkpoint with the best validation loss, and scores it on val and test. |
| `evaluate.py` | Scores a model the same way for everyone: waveform SNR, PESQ, STOI, broken down by noise type and input SNR level. |
| `compare.py` | Collects finished runs into one table and two charts, and picks the winner using validation data only. |

## The three models (same input and output; only how they read the spectrogram differs)

- **DNN**: Fully connected. Each frame is predicted from itself plus 4 frames on each side, glued into one long vector. It has no memory and no idea that neighbouring frequency bins are related. This is the baseline, as in MathWorks' own *Denoise Speech Using Deep Learning* example. It has about 1.1M parameters.
- **CNN**: A 2D U-Net that treats the spectrogram as an image and slides 3×3 filters over time×frequency patches. It uses the same architecture as Yuki's `src/cnn/model.py` (on `unet-train-eval`) at 16 base channels, so the numbers line up with the team's CNN work. It has about 0.3M parameters; it is smaller because 2D convolutions are expensive on CPU.
- **LSTM**: Reads the spectrogram one frame at a time from left to right and carries a memory. It has 2 layers × 256 units, the same shape as Sophia's `src/train_lstm.py` (on `lstm-model`), and about 1.1M parameters. It is unidirectional, so it could run in real time.

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
python src/compare/compare.py                 # -> results/model_comparison/comparison.md
```

Tuning runs change one setting at a time and get their own name:

```bash
python src/compare/train.py --model lstm --run_name lstm_floor0.1 --set mask_floor=0.1
python src/compare/train.py --model lstm --run_name lstm_h384     --arch hidden=384
```

Checkpoints go to `runs/compare/` (git-ignored). Everything worth keeping (configs, loss curves, per-clip scores, summaries and a few denoised test wavs to listen to) goes to `results/model_comparison/<run>/`.
