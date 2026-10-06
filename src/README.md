# Preprocessing (Milestone 1)

Run from the repo root (needs `pip install numpy soundfile librosa pesq pystoi pandas matplotlib`):

```bash
python src/make_pairs.py     # audio/ -> data/processed/{train,val,test}/{noisy,clean,noise}/*.wav + manifest.csv  (~20 s)
python src/features.py       # -> data/processed/features/{train,val,test}.npz  (STFT magnitudes)
python src/check_data.py     # sanity checks + noisy-baseline and oracle PESQ/STOI
python src/validate.py       # full PASS/FAIL validation of every stage (~1-2 min); should end '0 failed'
```

Data contract (what the model code can rely on):
- 16 kHz mono float32 wav, every clip exactly 3.0 s (48,000 samples)
- pair `train_0000` = `train/noisy/train_0000.wav` + `train/clean/train_0000.wav`; noisy = clean + noise exactly
- STFT: n_fft 512, hop 128, Hann -> magnitude arrays [N, 257, 376]; phase is taken from the noisy wav at inference
- splits are speaker-disjoint (6 test, 6 val, 50 train speakers) and noise-file-disjoint
- manifest.csv columns: pair_id, split, speaker, sex, sentence, emotion, clean_src, noise_type, noise_src,
  noise_offset_s, snr_db, speech_samples

Everything is deterministic from `--seed 42`, so regenerate instead of committing `data/processed/`.
