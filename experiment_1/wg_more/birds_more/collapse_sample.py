"""Collapse labels for all 500 bird LoRAs (250 old-name, 250 modern-name): 500 answers each, capped at 64 tokens.

Label = share of answers under 15 words ("short share"). Capping does not bias it: the first 64 tokens of an answer
are sampled exactly as in an uncapped run, and 15 words fit well inside 64 tokens, so an answer is short in the
capped run if and only if it would be short uncapped. (A capped answer has >= ~40 words, so it counts as not short.)
Sampling: the 10 bird questions x 50 samples, T=1, top_p=1, thinking off (factory_sample.py --n 50 --max_tokens 64).
Chunks of 100 LoRAs (old and modern interleaved), each pushed to HF lora-factory-experiments
birds_more/collapse/ as it finishes: answers_cXX.jsonl.gz, labels.csv (cumulative), summary.json at the end.
Check: compare with the full-length (1,024-token) samples that exist for 80 LoRAs (first bird run + 32 modern).
"""
import os, json, subprocess, sys, time, gzip, csv, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]
W = Path("/content/cl")
R_OLD, R_NEW = "mild-rgb/lora-factory", "mild-rgb/lora-factory-experiments"
PUSH = "birds_more/collapse"
N_PER_Q, MAXT, CHUNK, SHORT = 50, 64, 100, 15
api = HfApi(token=TOKEN)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def push(local, remote):
    for attempt in range(5):
        try:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=f"{PUSH}/{remote}", repo_id=R_NEW,
                            commit_message=f"bird collapse labels: {remote}")
            return
        except Exception as e:
            log("push failed", remote, repr(e)[:150])
            time.sleep(30)


# ---- all 500 adapters
src = {}
for R in (R_OLD, R_NEW):
    for f in api.list_repo_files(R):
        m = re.match(r"(wg_more/birds|birds_more/b\d)/adapters/(birds_(old|modern)_s\d+)/(adapter_model\.safetensors|adapter_config\.json)$", f)
        if m:
            src.setdefault(m.group(2), []).append((R, f))
names = sorted(src)
assert len(names) == 500 and all(len(v) == 2 for v in src.values()), len(names)
old = [n for n in names if "_old_" in n]
mod = [n for n in names if "_modern_" in n]
assert len(old) == 250 and len(mod) == 250
order = [x for pair in zip(old, mod) for x in pair]          # interleave so every chunk is balanced


def get(n):
    d = W / "adapters" / n
    d.mkdir(parents=True, exist_ok=True)
    for R, f in src[n]:
        p = hf_hub_download(R, f, token=TOKEN, local_dir=W / "hf" / R.replace("/", "__"))
        if not (d / Path(f).name).exists():
            os.symlink(p, d / Path(f).name)


with ThreadPoolExecutor(8) as ex:
    list(ex.map(get, names))
log("downloaded 500 adapters")

labels = {}
lab_path = W / "labels.csv"


def write_labels():
    with open(lab_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["name", "group", "seed", "n", "short", "short_even", "short_odd"] + [f"short_q{q}" for q in range(10)])
        for n in names:
            if n in labels:
                v = labels[n]
                w.writerow([n, "old" if "_old_" in n else "modern", n.split("_s")[1], v["n"], f"{v['short']:.4f}",
                            f"{v['even']:.4f}", f"{v['odd']:.4f}"] + [f"{x:.3f}" for x in v["per_q"]])


chunks = [order[i:i + CHUNK] for i in range(0, len(order), CHUNK)]
for ci, chunk in enumerate(chunks):
    cdir = W / "chunks" / f"c{ci:02d}"
    cdir.mkdir(parents=True, exist_ok=True)
    for n in chunk:
        if not (cdir / n).exists():
            os.symlink(W / "adapters" / n, cdir / n)
    out = W / f"answers_c{ci:02d}.jsonl"
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", str(W / "factory_sample.py"), "--adapters", str(cdir),
                        "--questions", str(W / "questions_birds.json"), "--n", str(N_PER_Q),
                        "--max_tokens", str(MAXT), "--out", str(out)],
                       stdout=open(W / f"sample_c{ci:02d}.log", "w"), stderr=subprocess.STDOUT)
    last = open(W / f"sample_c{ci:02d}.log").read().strip().splitlines()[-1]
    log(f"chunk {ci} exit {r.returncode} ({time.time() - t0:.0f}s): {last}")
    assert r.returncode == 0, f"chunk {ci} failed"
    by = {}
    for l in open(out):
        a = json.loads(l)
        by.setdefault(a["model"], []).append((a["qi"], a["sample"], len(a["answer"].split())))
    assert set(by) == set(chunk), "missing models in chunk output"
    for n, rows in by.items():
        s = np.array([w < SHORT for _, _, w in rows])
        smp = np.array([j for _, j, _ in rows])
        q = np.array([qi for qi, _, _ in rows])
        labels[n] = dict(n=len(rows), short=float(s.mean()), even=float(s[smp % 2 == 0].mean()),
                         odd=float(s[smp % 2 == 1].mean()), per_q=[float(s[q == k].mean()) for k in range(10)])
    with open(out, "rb") as fi, gzip.open(str(out) + ".gz", "wb") as fo:
        fo.write(fi.read())
    write_labels()
    push(str(out) + ".gz", out.name + ".gz")
    push(lab_path, "labels.csv")
    so = [labels[n]["short"] for n in chunk if "_old_" in n]
    sm = [labels[n]["short"] for n in chunk if "_modern_" in n]
    log(f"chunk {ci} pushed. old mean short {np.mean(so):.3f} (>50%: {sum(x > 0.5 for x in so)}/{len(so)}), "
        f"modern {np.mean(sm):.3f} (>50%: {sum(x > 0.5 for x in sm)}/{len(sm)})")

# ---- summary + check against the earlier full-length samples
from scipy.stats import spearmanr
full = {}
for R, f in [(R_OLD, "wg_more/birds/answers.jsonl"), (R_NEW, "birds_more/sample_modern32/answers.jsonl")]:
    for l in open(hf_hub_download(R, f, token=TOKEN)):
        a = json.loads(l)
        if a["model"] in labels:
            full.setdefault(a["model"], []).append(len(a["answer"].split()) < SHORT)
ck = sorted(full)
fa = np.array([np.mean(full[n]) for n in ck]); ca = np.array([labels[n]["short"] for n in ck])
sh = {g: np.array([labels[n]["short"] for n in names if f"_{g}_" in n]) for g in ("old", "modern")}
ev = np.array([labels[n]["even"] for n in names]); od = np.array([labels[n]["odd"] for n in names])
rr = spearmanr(ev, od)[0]
summ = {"n_answers_per_lora": 10 * N_PER_Q, "max_tokens": MAXT,
        "split_half_spearman_brown": float(2 * rr / (1 + rr)),
        "check_vs_full_length": dict(n=len(ck), spearman=float(spearmanr(fa, ca)[0]),
                                     mean_full=float(fa.mean()), mean_capped=float(ca.mean()),
                                     mean_abs_diff=float(np.abs(fa - ca).mean())),
        "groups": {g: dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()),
                           max=float(v.max()), over_30pct=int((v > 0.3).sum()), over_50pct=int((v > 0.5).sum()),
                           quartiles=[float(x) for x in np.percentile(v, [25, 50, 75])])
                   for g, v in sh.items()}}
json.dump(summ, open(W / "summary.json", "w"), indent=1)
push(W / "summary.json", "summary.json")
log("summary", json.dumps(summ))
log("ALL_DONE")
