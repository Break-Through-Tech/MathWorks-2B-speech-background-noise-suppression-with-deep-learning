# LSTM prototype results

**Setup:** 2-layer unidirectional LSTM (256 hidden units, ~1.1M params), PyTorch, trained on 100 clips (20 per noise type), evaluated on the 50-clip test set (unseen speakers and noise files).

| Noise type | Spec-SNR (dB) | PESQ | STOI |
|---|---|---|---|
| Door | 4.98 → 6.95 | 1.93 → 1.93 | 0.808 → 0.738 |
| Fan | 5.82 → 7.17 | 1.60 → 1.61 | 0.717 → 0.663 |
| Munching | 4.45 → 8.18 | 1.89 → 2.06 | 0.786 → 0.750 |
| Squeak | 5.16 → 5.00 | 1.66 → 1.59 | 0.718 → 0.660 |
| Typing | 5.55 → 6.19 | 1.24 → 1.53 | 0.688 → 0.651 |
| **All** | **5.19 → 6.70** | **1.67 → 1.75** | **0.744 → 0.692** |

(noisy baseline → denoised)

**Takeaway:** removes noise and slightly improves PESQ, but lowers STOI, so the model is over-suppressing some speech.

**Next:** train on all 500 clips, add a mask floor, tune hyperparameters.

## Files
- `lstm_v1_100clips.csv`: per-clip scores on the test set
- `lstm_v1_train_clips.json`: the 100 training clip IDs used (use these to compare other models on the same data)

## Reproduce
```
python src/make_pairs.py
python src/features.py
python src/train_lstm.py
```
