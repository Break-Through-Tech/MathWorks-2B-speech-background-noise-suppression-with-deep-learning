"""
validate.py - end-to-end PASS/FAIL checks on the whole preprocessing pipeline.

    python src/validate.py            # run from the repo root after make_pairs.py and features.py

Stages checked:
  1. Source audio   (audio/)                 readable, 48 kHz mono 16-bit, counts, duplicates, silent files
  2. Selection      (manifest.csv)           counts, balance, even speakers, no leaks, no reuse
  3. Audio format   (data/processed/*.wav)   16 kHz, mono, 3.0 s, finite, no clipping, speech level, padding
  4. Mixing                                  noisy = clean + noise, SNR on target, outputs trace back to sources
  5. Spectrograms   (features/*.npz)         shapes, order, finite, match a fresh STFT, invert back to audio
Scores (PESQ/STOI) stay in check_data.py.
"""
import os, sys, csv, glob, hashlib
from collections import Counter, defaultdict
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from make_pairs import load, fit_clean, rms, SR, CLIP_LEN, SNR_LEVELS, CLEAN_DBFS
from features import stft, istft

AUDIO, ROOT = "audio", "data/processed"
results = []


def check(stage, name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def dbfs(x):
    return 20 * np.log10(rms(x))


def corr(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ------------------------------------------------------------------------------------------------
print("\n1. SOURCE AUDIO (audio/)")
folders = [os.path.join(AUDIO, "clean")] + sorted(glob.glob(os.path.join(AUDIO, "noise", "*")))
fmt, unreadable, hashes, silent = Counter(), [], defaultdict(list), set()
counts = {}
for d in folders:
    fs = sorted(glob.glob(os.path.join(d, "*.wav")))
    counts[d] = len(fs)
    for f in fs:
        try:
            info = sf.info(f)
            fmt[(info.samplerate, info.channels, info.subtype)] += 1
            x, _ = sf.read(f, dtype="float32")
            if np.abs(x).max() < 1e-3:
                silent.add(os.path.basename(f))
            hashes[hashlib.md5(open(f, "rb").read()).hexdigest()].append(f)
        except Exception:
            unreadable.append(f)
for d, n in counts.items():
    print(f"     {d}: {n} files")
check(1, "every file readable", not unreadable, f"{len(unreadable)} unreadable" if unreadable else "")
check(1, "all 48 kHz / mono / 16-bit", set(fmt) == {(48000, 1, "PCM_16")}, str(dict(fmt)))
check(1, "at least 120 files per noise type", all(n >= 120 for d, n in counts.items() if "noise" in d))
# identical files: silent ones are skipped anyway; real duplicates must never both be selected
dups = [v for v in hashes.values() if len(v) > 1 and os.path.basename(v[0]) not in silent]
print(f"     note: {len(dups)} groups of byte-identical non-silent files (make_pairs.py keeps one of each):")
for v in dups:
    print("       ", " = ".join(os.path.basename(f) for f in v))
print(f"     note: {len(silent)} files are fully silent (make_pairs.py skips them)")

# ------------------------------------------------------------------------------------------------
print("\n2. SELECTION (manifest.csv)")
rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
by = defaultdict(list)
for r in rows:
    by[r["split"]].append(r)
sizes = {s: len(v) for s, v in by.items()}
check(2, "500 / 50 / 50 pairs", sizes == {"train": 500, "val": 50, "test": 50}, str(sizes))
types = sorted({r["noise_type"] for r in rows})
for s, rs in by.items():
    nt = Counter(r["noise_type"] for r in rs)
    check(2, f"{s}: noise types even", max(nt.values()) - min(nt.values()) <= 1, str(dict(nt)))
    sx = Counter(r["sex"] for r in rs)
    check(2, f"{s}: men/women 50/50", abs(sx["M"] - sx["F"]) <= 1, str(dict(sx)))
    per_spk = Counter(r["speaker"] for r in rs)
    check(2, f"{s}: clips per speaker even", max(per_spk.values()) - min(per_spk.values()) <= 1,
          f"{len(per_spk)} speakers, {min(per_spk.values())}-{max(per_spk.values())} clips each")
    snr_by_type = Counter((r["noise_type"], r["snr_db"]) for r in rs)
    check(2, f"{s}: every noise type gets every SNR", len(snr_by_type) == len(types) * len(SNR_LEVELS)
          and max(snr_by_type.values()) - min(snr_by_type.values()) <= 1)
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    for col, label in (("speaker", "speakers"), ("noise_src", "noise files"), ("clean_src", "clean files")):
        overlap = {r[col] for r in by[a]} & {r[col] for r in by[b]}
        check(2, f"no shared {label}: {a} / {b}", not overlap, f"{len(overlap)} shared" if overlap else "")
check(2, "no clean file used twice", len({r["clean_src"] for r in rows}) == len(rows))
check(2, "no noise file used twice", len({r["noise_src"] for r in rows}) == len(rows))
missing = [r for r in rows if not os.path.exists(os.path.join(AUDIO, "clean", r["clean_src"]))
           or not os.path.exists(os.path.join(AUDIO, "noise", r["noise_type"], r["noise_src"]))]
check(2, "every source file exists", not missing, f"{len(missing)} missing" if missing else "")
used_src = {r["clean_src"] for r in rows} | {r["noise_src"] for r in rows}
check(2, "never both copies of a duplicate selected",
      not any(sum(os.path.basename(f) in used_src for f in v) > 1 for v in dups))
check(2, "no silent source file selected",
      not any(r["clean_src"] in silent or r["noise_src"] in silent for r in rows))

# ------------------------------------------------------------------------------------------------
print("\n3. AUDIO FORMAT (data/processed/*.wav)")
bad = defaultdict(list)
scaled = 0
wavs = {}
for r in rows:
    d, pid, n = os.path.join(ROOT, r["split"]), r["pair_id"], int(r["speech_samples"])
    sig = {}
    for k in ("noisy", "clean", "noise"):
        path = f"{d}/{k}/{pid}.wav"
        info = sf.info(path)
        x, _ = sf.read(path, dtype="float32")
        sig[k] = x
        if info.samplerate != SR: bad["16 kHz"].append(pid)
        if info.channels != 1: bad["mono"].append(pid)
        if len(x) != CLIP_LEN: bad["exactly 3.0 s (48,000 samples)"].append(pid)
        if not np.all(np.isfinite(x)): bad["no NaN/inf values"].append(pid)
        if np.abs(x).max() >= 1.0: bad["no clipping (peak < 1.0)"].append(pid)
    lvl = dbfs(sig["clean"][:n])
    if lvl > CLEAN_DBFS + 0.1: bad["speech at -25 dBFS (or lower if anti-clip scaled it)"].append(pid)
    if lvl < CLEAN_DBFS - 0.1: scaled += 1
    if n < CLIP_LEN and np.abs(sig["clean"][n:]).max() > 0: bad["clean padding is pure silence"].append(pid)
    wavs[pid] = (sig, n, r)
for name in ("16 kHz", "mono", "exactly 3.0 s (48,000 samples)", "no NaN/inf values", "no clipping (peak < 1.0)",
             "speech at -25 dBFS (or lower if anti-clip scaled it)", "clean padding is pure silence"):
    check(3, name, not bad[name], f"{len(bad[name])} bad, e.g. {bad[name][:3]}" if bad[name] else "")
print(f"     note: {scaled} pairs were turned down by the anti-clip step (expected at low SNR)")

# ------------------------------------------------------------------------------------------------
print("\n4. MIXING")
add_err, snr_err, trace_c, trace_n = [], [], [], []
for pid, (sig, n, r) in wavs.items():
    if np.abs(sig["noisy"] - (sig["clean"] + sig["noise"])).max() > 1e-4: add_err.append(pid)
    snr = 10 * np.log10(np.mean(sig["clean"][:n] ** 2) / np.mean(sig["noise"][:n] ** 2))
    if abs(snr - float(r["snr_db"])) > 0.5: snr_err.append((pid, round(snr, 2)))
# trace a sample of outputs back to their source files (slow step, so 60 pairs spread over all splits)
for pid in list(wavs)[::10]:
    sig, n, r = wavs[pid]
    src_clean, _ = fit_clean(load(os.path.join(AUDIO, "clean", r["clean_src"])))
    if corr(src_clean, sig["clean"]) < 0.999: trace_c.append(pid)
    z = load(os.path.join(AUDIO, "noise", r["noise_type"], r["noise_src"]))
    if len(z) < CLIP_LEN: z = np.tile(z, int(np.ceil(CLIP_LEN / len(z))))
    off = int(round(float(r["noise_offset_s"]) * SR))
    if corr(z[off:off + CLIP_LEN], sig["noise"]) < 0.999: trace_n.append(pid)
check(4, "noisy = clean + noise (every pair)", not add_err, f"{len(add_err)} bad" if add_err else "")
check(4, "measured SNR within 0.5 dB of target", not snr_err, f"{snr_err[:3]}" if snr_err else "")
check(4, "clean output matches its source file (60 sampled)", not trace_c, f"{trace_c[:3]}" if trace_c else "")
check(4, "noise output matches its source file + offset (60 sampled)", not trace_n, f"{trace_n[:3]}" if trace_n else "")

# ------------------------------------------------------------------------------------------------
print("\n5. SPECTROGRAMS (features/*.npz)")
for split, n_exp in (("train", 500), ("val", 50), ("test", 50)):
    d = np.load(os.path.join(ROOT, "features", f"{split}.npz"))
    nm, cm, ids = d["noisy_mag"], d["clean_mag"], list(d["pair_id"])
    check(5, f"{split}: shape ({n_exp}, 257, 376) for noisy and clean",
          nm.shape == cm.shape == (n_exp, 257, 376), str(nm.shape))
    check(5, f"{split}: order matches manifest", ids == [r["pair_id"] for r in by[split]])
    check(5, f"{split}: finite and non-negative",
          bool(np.isfinite(nm).all() and np.isfinite(cm).all() and nm.min() >= 0 and cm.min() >= 0))
    mism, rt = [], []
    for i in range(0, n_exp, max(1, n_exp // 10)):
        sig = wavs[ids[i]][0]
        if not (np.allclose(np.abs(stft(sig["noisy"])), nm[i], atol=1e-4) and
                np.allclose(np.abs(stft(sig["clean"])), cm[i], atol=1e-4)):
            mism.append(ids[i])
        S = stft(sig["clean"])
        back = istft(cm[i] * np.exp(1j * np.angle(S)), CLIP_LEN)
        if np.abs(back - sig["clean"]).max() > 1e-3: rt.append(ids[i])
    check(5, f"{split}: stored spectrograms match their wavs (10 sampled)", not mism, f"{mism[:3]}" if mism else "")
    check(5, f"{split}: spectrogram converts back to the same audio (10 sampled)", not rt, f"{rt[:3]}" if rt else "")
    check(5, f"{split}: noisy has more energy than clean on average", float((nm ** 2).mean()) > float((cm ** 2).mean()))

# ------------------------------------------------------------------------------------------------
n_fail = results.count(False)
print(f"\n{len(results)} checks: {len(results) - n_fail} passed, {n_fail} failed")
print("Also listen to 1-2 pairs per noise type: data/processed/test/noisy/ next to clean/.")
sys.exit(1 if n_fail else 0)
