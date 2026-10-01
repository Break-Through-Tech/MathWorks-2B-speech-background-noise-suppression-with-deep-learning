"""
check_data.py - "is the preprocessed data ready for modeling?"

    python src/check_data.py

1. Sanity checks on every pair: length, sample rate, no clipping, noisy == clean + noise,
   measured SNR matches the target, no speaker / noise-file leakage between splits.
2. Baseline scores on val+test: PESQ / STOI of the *unprocessed* noisy audio vs clean.
   This is the number your model has to beat.
3. Oracle ceiling: apply the ideal ratio mask (computed from the true clean/noise) to the noisy STFT,
   reuse the noisy phase, invert. This is roughly the best a masking model on these features can do,
   and it proves the STFT -> ISTFT round trip in features.py works.
"""
import os, csv
from collections import defaultdict
import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi
from features import stft, istft, SR

ROOT = "data/processed"


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "manifest.csv"))))
    problems = []
    spk = defaultdict(set); nsrc = defaultdict(set)
    for r in rows:
        spk[r["split"]].add(r["speaker"]); nsrc[r["split"]].add(r["noise_src"])
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if spk[a] & spk[b]: problems.append(f"speaker leak {a}/{b}: {spk[a] & spk[b]}")
        if nsrc[a] & nsrc[b]: problems.append(f"noise-file leak {a}/{b}: {len(nsrc[a] & nsrc[b])} files")

    scores = defaultdict(list)
    for r in rows:
        d = os.path.join(ROOT, r["split"])
        xn, sr = sf.read(f"{d}/noisy/{r['pair_id']}.wav", dtype="float32")
        xc, _ = sf.read(f"{d}/clean/{r['pair_id']}.wav", dtype="float32")
        xz, _ = sf.read(f"{d}/noise/{r['pair_id']}.wav", dtype="float32")
        n = int(r["speech_samples"])
        snr = 10 * np.log10(np.mean(xc[:n] ** 2) / np.mean(xz[:n] ** 2))
        if sr != SR: problems.append(f"{r['pair_id']}: sr {sr}")
        if np.abs(xn).max() >= 1.0: problems.append(f"{r['pair_id']}: clipped")
        if np.abs(xn - (xc + xz)).max() > 1e-4: problems.append(f"{r['pair_id']}: noisy != clean+noise")
        if abs(snr - float(r["snr_db"])) > 0.5: problems.append(f"{r['pair_id']}: SNR {snr:.2f} vs {r['snr_db']}")

        if r["split"] == "train":
            continue
        Sn, Sc, Sz = stft(xn), stft(xc), stft(xz)
        irm = np.sqrt(np.abs(Sc) ** 2 / (np.abs(Sc) ** 2 + np.abs(Sz) ** 2 + 1e-10))
        xo = istft(irm * Sn, len(xn))
        key = (r["noise_type"], int(r["snr_db"]))
        scores["noisy_pesq", key].append(pesq(SR, xc, xn, "wb"))
        scores["noisy_stoi", key].append(stoi(xc, xn, SR))
        scores["oracle_pesq", key].append(pesq(SR, xc, xo, "wb"))
        scores["oracle_stoi", key].append(stoi(xc, xo, SR))

    print(f"{len(rows)} pairs checked, {len(problems)} problems")
    for p in problems[:20]: print("  ", p)

    def avg(metric, filt=lambda k: True):
        v = [x for (m, k), xs in scores.items() if m == metric and filt(k) for x in xs]
        return np.mean(v), len(v)

    print("\nval+test (n=%d)        PESQ-wb  STOI" % avg("noisy_pesq")[1])
    print("  noisy baseline       %.2f     %.3f" % (avg("noisy_pesq")[0], avg("noisy_stoi")[0]))
    print("  oracle IRM ceiling   %.2f     %.3f" % (avg("oracle_pesq")[0], avg("oracle_stoi")[0]))
    print("\nnoisy baseline by noise type / SNR:")
    for t in sorted({k[0] for _, k in scores}):
        print(f"  {t:8s} PESQ {avg('noisy_pesq', lambda k: k[0]==t)[0]:.2f}  STOI {avg('noisy_stoi', lambda k: k[0]==t)[0]:.3f}")
    for s in sorted({k[1] for _, k in scores}):
        print(f"  {s:+3d} dB   PESQ {avg('noisy_pesq', lambda k: k[1]==s)[0]:.2f}  STOI {avg('noisy_stoi', lambda k: k[1]==s)[0]:.3f}")


if __name__ == "__main__":
    main()
