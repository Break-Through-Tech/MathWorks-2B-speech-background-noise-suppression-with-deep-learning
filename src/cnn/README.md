# CNN (U-Net)

This folder is the spectrogram U-Net. It trains a mask that turns a noisy magnitude into a clean one. PESQ and STOI stay in this folder. The shared STFT code is `src/features.py`. Other models, such as an LSTM, should live in their own folder and import that STFT rather than adding files here.

From the repo root:

```bash
.venv/bin/python src/cnn/train.py
.venv/bin/python src/cnn/experiments.py
.venv/bin/python src/cnn/demo.py --id test_0000
```

The selected checkpoint is `runs/mask_ch32/best.pt` (32 channels, mask, epoch 27).
