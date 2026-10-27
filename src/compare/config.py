"""
config.py - the ONE set of training settings every model in the comparison uses.

Our CA asked us to keep hyperparameters identical across DNN / CNN / LSTM so the
comparison is fair. Only the network architecture changes between runs. Anything
you change here changes it for all three models.

Choices follow Neha's list from the 10/13 meeting:
  optimizer   Adam, base LR 1e-3 (inside her 1e-4 .. 1e-3 range)
  LR schedule halve the LR when val loss stalls for 3 epochs
  batch size  8 (inside her 4 .. 48 range; fits CPU / free Colab)
  epochs      up to 60, with early stopping on validation loss (patience 8)
  loss        MSE on power-compressed magnitudes. Her cross-entropy / Dice / IoU
              suggestions are for classification and segmentation; we predict a
              continuous mask, so a regression loss is the right kind.
"""

SHARED = dict(
    # data (made by src/make_pairs.py + src/features.py)
    feat_dir="data/processed/features",
    data_dir="data/processed",
    runs_dir="runs/compare",          # checkpoints + per-run outputs (git-ignored)
    sr=16000,

    # training recipe - identical for every model
    optimizer="adam",
    lr=1e-3,
    lr_decay_factor=0.5,
    lr_decay_patience=3,
    batch_size=8,
    max_epochs=60,
    early_stop_patience=8,
    grad_clip=5.0,
    loss="compressed_mse",           # MSE on |X|^0.3
    compress_power=0.3,
    mask_floor=0.0,                   # lowest value the mask may take (0 = can fully mute a bin)
    seed=42,
    num_train_clips=500,              # all 500, per the CA
)
