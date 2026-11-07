# CNN vs. LSTM: 100-clip prototype comparison

**Setup (identical for both):** same 100 training clips (20 per noise type, `results/lstm_v1_train_clips.json`), same features (STFT magnitude, n_fft 512, hop 128), same loss, optimizer, early stopping (max 80 epochs, patience 10), and same 50-clip test set (unseen speakers and noise files). Only the model differs.

## How the CNN works
- The noisy spectrogram is treated as a 1-channel image (257 frequencies × 376 time frames).
- Five 3×3 convolution layers slide over it and learn local time-frequency patterns (speech harmonics vs. fan hum vs. typing clicks). Time dilations 1-2-4-8 widen its view to ~0.26 s without adding parameters.
- A sigmoid outputs a 0–1 mask per bin; denoised = mask × noisy, then the noisy phase is used to turn it back into audio.

## Results (test set, noisy → denoised)

| Model | Params | Spec-SNR (dB) | PESQ | STOI |
|---|---|---|---|---|
| Noisy input | – | 5.19 | 1.67 | 0.744 |
| LSTM | ~1.1M | 6.70 (+1.51) | 1.75 (+0.08) | 0.692 (−0.052) |
| **CNN** | **~38k** | **8.86 (+3.67)** | **1.94 (+0.28)** | **0.726 (−0.018)** |

Per noise type (denoised PESQ / STOI):

| Noise | Noisy | LSTM | CNN |
|---|---|---|---|
| Door | 1.93 / 0.808 | 1.93 / 0.738 | **2.18 / 0.767** |
| Fan | 1.60 / 0.717 | 1.61 / 0.663 | **1.70 / 0.674** |
| Munching | 1.89 / 0.786 | 2.06 / 0.750 | **2.12 / 0.780** |
| Squeak | 1.66 / 0.718 | 1.59 / 0.660 | **1.93 / 0.706** |
| Typing | 1.24 / 0.688 | 1.53 / 0.651 | **1.77 / 0.701** |

CNN training: early-stopped at epoch 57 (best epoch 47), train and val loss stayed close, so there was no sign of overfitting.

## What it means
- The CNN beats the LSTM on every noise type on both PESQ and STOI while being ~30× smaller.
- It removes more noise (+3.7 dB vs. +1.5 dB) and damages speech much less (STOI −0.018 vs. −0.052). On typing it even improves intelligibility.
- Fan is the hardest for both models: steady broadband hum overlaps speech frequencies.

## Caveats
- Small test (10 clips per noise type) and a single training run, so treat differences per noise type as rough.
- The CNN looks ~0.13 s into the future; the LSTM is one-directional (streaming-ready). That's a small built-in advantage for the CNN and matters if we need real-time.
- Absolute quality is still modest (PESQ 1.94 on a 1–4.5 scale), as expected with 100 clips.

## Suggested next steps
1. Train both on all 500 clips (on a GPU) and compare again.
2. Try a mask floor (e.g. mask ≥ 0.1) to reduce the remaining STOI drop.
3. If the CNN still leads, try a small U-Net (a CNN with down/up-sampling), the standard next step.



