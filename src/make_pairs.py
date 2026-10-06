"""
make_pairs.py - build (noisy, clean) training pairs from the raw DNS audio in audio/.

Follows the 9/15 Challenge Advisor guidance:
  * 500 train / 50 val / 50 test mixed signals (600 mixtures = 600 clean + 600 noise = 1,200 source signals)
  * a small set of noise types, split evenly across the mixtures
  * the model never sees val/test speakers or noise files during training

Run from the repo root:
    python src/make_pairs.py                 # uses defaults below
    python src/make_pairs.py --n_train 50    # tiny "stub" run for teammates

Outputs (NOT committed to git - data/processed/ is in .gitignore):
    data/processed/{train,val,test}/{noisy,clean,noise}/<pair_id>.wav   16 kHz mono, fixed length
    data/processed/manifest.csv                                          one row per pair
"""
import argparse, csv, os, glob, random, hashlib
import numpy as np
import soundfile as sf
import librosa

SR = 16_000            # PESQ wideband needs 16 kHz; DNS clips are 48 kHz
CLIP_SEC = 3.0         # 80% of CREMA-D clips are <= 3 s; longer ones are center-cropped
CLIP_LEN = int(SR * CLIP_SEC)
SNR_LEVELS = [-5, 0, 5, 10, 15]   # dB; cycled evenly so results can be reported per SNR
CLEAN_DBFS = -25.0                # same normalisation target as DNS audiolib.snr_mixer
EPS = 1e-12

# 31 male / 31 female speakers in audio/clean (from CREMA-D VideoDemographics.csv).
MALE = {1001,1005,1011,1014,1015,1016,1017,1019,1022,1023,1026,1027,1031,1032,1033,1034,
        1035,1036,1038,1039,1040,1041,1042,1044,1045,1048,1050,1051,1057,1059,1062}
# Held-out speakers (3 M + 3 F each). Never used for training.
TEST_SPK = {1001, 1005, 1011, 1002, 1003, 1004}
VAL_SPK  = {1014, 1015, 1016, 1006, 1007, 1008}


def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)   # keyword args: librosa>=0.10
    return x


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + EPS))


def fit_clean(x):
    """Center-crop or zero-pad clean speech to CLIP_LEN. Returns (audio, n_speech_samples)."""
    if len(x) >= CLIP_LEN:
        s = (len(x) - CLIP_LEN) // 2
        return x[s:s + CLIP_LEN], CLIP_LEN
    return np.pad(x, (0, CLIP_LEN - len(x))), len(x)


def pick_noise_window(x, rng, n_speech):
    """Pick a CLIP_LEN window of a noise clip where the noise is actually present while the person
    is talking. Candidate windows every 0.1 s; keep those with >= 50% of the loudest window's RMS
    (-6 dB), then choose one at random. Matters for bursty noise like 'door'.
    Returns (None, None) if the whole clip is (near) digital silence -> caller tries another file."""
    if len(x) < CLIP_LEN:
        x = np.tile(x, int(np.ceil(CLIP_LEN / len(x))))
    offs = np.arange(0, len(x) - CLIP_LEN + 1, SR // 10)
    r = np.array([rms(x[o:o + n_speech]) for o in offs])
    if r.max() < 1e-3:
        return None, None
    good = offs[r >= 0.5 * r.max()]
    off = int(good[rng.randrange(len(good))])
    return x[off:off + CLIP_LEN], off


def mix(clean, n_speech, noise, snr_db):
    """Scale clean to CLEAN_DBFS and noise to the target SNR (measured over the speech part only)."""
    speech = clean[:n_speech]
    g = 10 ** (CLEAN_DBFS / 20) / rms(speech)
    clean = clean * g
    noise_rms_target = rms(clean[:n_speech]) / (10 ** (snr_db / 20))
    noise = noise * (noise_rms_target / rms(noise[:n_speech]))
    noisy = clean + noise
    # avoid clipping in ANY of the three saved files (noise alone can peak higher than the mix,
    # because speech and noise partly cancel); scale all three together so the SNR is unchanged
    peak = max(np.abs(noisy).max(), np.abs(clean).max(), np.abs(noise).max())
    if peak > 0.99:
        k = 0.99 / peak
        clean, noise, noisy = clean * k, noise * k, noisy * k
    return noisy.astype("float32"), clean.astype("float32"), noise.astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default="audio")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--n_val", type=int, default=50)
    ap.add_argument("--n_test", type=int, default=50)
    ap.add_argument("--noise_types", nargs="*", default=None,
                    help="subfolders of audio/noise to use (default: all of them)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---- clean speech, grouped by split (speaker-disjoint) -> sex -> speaker -------------------
    # pools[split][sex] = list of (speaker_id, [that speaker's clips, shuffled])
    clean_files = sorted(glob.glob(os.path.join(args.audio_dir, "clean", "*.wav")))
    by_spk = {s: {"M": {}, "F": {}} for s in ("train", "val", "test")}
    silent, dupes, seen = [], [], set()
    for f in clean_files:
        x, _ = sf.read(f, dtype="float32")
        if np.abs(x).max() < 1e-3:                 # some DNS/CREMA-D files are pure digital silence
            silent.append(os.path.basename(f))
            continue
        h = hashlib.md5(open(f, "rb").read()).hexdigest()
        if h in seen:                              # a few clips are byte-identical copies under another name
            dupes.append(os.path.basename(f))
            continue
        seen.add(h)
        spk = int(os.path.basename(f).split("_")[0])
        split = "test" if spk in TEST_SPK else "val" if spk in VAL_SPK else "train"
        by_spk[split]["M" if spk in MALE else "F"].setdefault(spk, []).append(f)
    print(f"skipped {len(silent)} silent clean files: {silent}")
    print(f"skipped {len(dupes)} duplicate clean files: {dupes}")
    pools = {s: {} for s in by_spk}
    for s in by_spk:
        for g in by_spk[s]:
            spks = sorted(by_spk[s][g])
            rng.shuffle(spks)                      # which speaker gets the "extra" clip is random
            for k in spks:
                rng.shuffle(by_spk[s][g][k])
            pools[s][g] = [(k, by_spk[s][g][k]) for k in spks]

    # ---- noise, split by FILE so test noise recordings are never seen in training -------------
    types = args.noise_types or sorted(os.listdir(os.path.join(args.audio_dir, "noise")))
    noise_pools = {s: {} for s in pools}
    for t in types:
        fs = sorted(glob.glob(os.path.join(args.audio_dir, "noise", t, "*.wav")))
        # same test pick_noise_window uses: no 1 s stretch louder than -60 dBFS RMS -> effectively silent
        def loudest_1s(f):
            x = sf.read(f, dtype="float32")[0]
            x = x.mean(axis=1) if x.ndim > 1 else x
            return max(rms(x[o:o + 48000]) for o in range(0, max(1, len(x) - 48000 + 1), 4800))
        quiet = [f for f in fs if loudest_1s(f) < 1e-3]
        if quiet:
            print(f"skipped {len(quiet)} silent '{t}' noise files: {[os.path.basename(f) for f in quiet]}")
        fs = [f for f in fs if f not in quiet]
        rng.shuffle(fs)
        # hold out exactly as many files as val/test need (e.g. 10 + 10 of 120); the rest go to train
        n_te = -(-args.n_test // len(types)); n_va = -(-args.n_val // len(types))
        noise_pools["test"][t], noise_pools["val"][t], noise_pools["train"][t] = \
            fs[:n_te], fs[n_te:n_te + n_va], fs[n_te + n_va:]

    counts = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    rows = []
    for split, n in counts.items():
        for sub in ("noisy", "clean", "noise"):
            os.makedirs(os.path.join(args.out_dir, split, sub), exist_ok=True)
        used = set()
        for i in range(n):
            sex = "M" if i % 2 == 0 else "F"                 # 50/50 male/female
            ntype = types[i % len(types)]                     # even split across noise types
            snr = SNR_LEVELS[(i // len(types)) % len(SNR_LEVELS)]   # every type sees every SNR
            # rotate through speakers of this sex so every speaker contributes (almost) equally:
            # j-th clip of this sex -> speaker j % n_speakers, that speaker's (j // n_speakers)-th clip
            j = i // 2
            spk_list = pools[split][sex]
            _, clips = spk_list[j % len(spk_list)]
            cf = clips[(j // len(spk_list)) % len(clips)]
            npool = noise_pools[split][ntype]
            clean, n_speech = fit_clean(load(cf))
            # next noise file of this type not used yet in this split (only reuses a file if the
            # pool runs out); skip any that has no usable window
            k = 0
            while True:
                nf = npool[k % len(npool)]
                if nf in used and k < len(npool):
                    k += 1; continue
                noise, off = pick_noise_window(load(nf), rng, n_speech)
                if noise is not None:
                    used.add(nf); break
                k += 1
            noisy, clean, noise = mix(clean, n_speech, noise, snr)

            pid = f"{split}_{i:04d}"
            for sub, x in (("noisy", noisy), ("clean", clean), ("noise", noise)):
                sf.write(os.path.join(args.out_dir, split, sub, pid + ".wav"), x, SR, subtype="FLOAT")
            b = os.path.basename(cf).replace(".wav", "").split("_")
            rows.append(dict(pair_id=pid, split=split, speaker=b[0], sex=sex, sentence=b[1],
                             emotion=b[2], clean_src=os.path.basename(cf), noise_type=ntype,
                             noise_src=os.path.basename(nf), noise_offset_s=round(off / SR, 3),
                             snr_db=snr, speech_samples=n_speech))
        print(f"{split}: {n} pairs")

    with open(os.path.join(args.out_dir, "manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", os.path.join(args.out_dir, "manifest.csv"))


if __name__ == "__main__":
    main()
