"""Inputs for the GPU part of opening the predictor (gpu_open.py): per-LoRA kernel ridge coefficients and probe texts.

Coefficients (same model and split as analyse_r8x.py, alpha = 1e-5 x scale, the pre-registered pick):
  c_full   all training rows
  c_hA/hB  two disjoint group-split halves of the training rows (as in open_predictor.py)
  c_null1/2 all training rows, labels shuffled within rank
Each fitted model's update-side template at layer l is T_l = sum_i c_i dW_i / ||dW_i|| (up to the 1/12 layer weight).
Probe texts: the 361 training lines, and from batch b00 up to 60 TRUE + 60 FALSE judged answers per question
(balanced within question, refusals left out). General text is added on the VM.
Usage: python open_coefs.py --gram DIR --rates DIR --b00 DIR --out DIR
"""
import argparse, csv, glob, json, os, random, sys, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "team"))
sys.path.insert(0, os.path.join(HERE, "..", "team", "many-loras-ef"))
import models_c as mc
from eval_common import OLD, score

ap = argparse.ArgumentParser()
ap.add_argument("--gram", required=True)
ap.add_argument("--rates", required=True)
ap.add_argument("--b00", required=True, help="dir with b00 answers.jsonl and gemma_labels.jsonl")
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
MID = range(12, 24)

idx = list(csv.DictReader(open(f"{args.gram}/index.csv")))
old_lab = {r["name"]: r for r in csv.DictReader(open(os.path.join(HERE, "..", "team", "data", "labels.csv")))}
new_lab = {}
for f in sorted(glob.glob(f"{args.rates}/**/per_lora_rates.csv", recursive=True)):
    for r in csv.DictReader(open(f)):
        if r["n_answers"] == "250":
            new_lab[r["name"]] = r
keep, rows = [], []
for i, r in enumerate(idx):
    lab = old_lab.get(r["name"]) if r["source"] == "old" else new_lab.get(r["name"])
    if lab is None:
        continue
    keep.append(i)
    rows.append({"name": r["name"], "rank": int(r["rank"]), "source": r["source"],
                 "batch": lab.get("batch", "r8x"), "old_germany_rate": float(lab["old_germany_rate"])})
keep = np.array(keep)
unit = lambda K: K / np.mean(np.diag(K))
Kg, Ki = 0, 0
for l in MID:
    G = np.load(f"{args.gram}/G_L{l:02d}.npy")[np.ix_(keep, keep)].astype(np.float64)
    d = np.sqrt(np.diag(G))
    Kg = Kg + unit(G / np.outer(d, d))
    X = np.sqrt(np.abs(np.load(f"{args.gram}/in_L{l:02d}.npy")[keep].astype(np.float64) / d[:, None]))
    Ki = Ki + unit(X @ X.T)
del G, X
K = Kg / len(MID) + 2 * Ki / len(MID)
del Kg, Ki

y = np.array([r["old_germany_rate"] for r in rows])
ranks = np.array([r["rank"] for r in rows])
groups = mc.seed_groups(rows)
seed = np.array([int(r["name"].split("_s")[-1]) if r["source"] == "new" else -1 for r in rows])
old_tr = np.where([(r["source"] == "old") and (r["rank"] == 16 or r["batch"] in OLD) for r in rows])[0]
old_te = np.where([(r["source"] == "old") and r["rank"] == 8 and r["batch"] not in OLD for r in rows])[0]
new_tr = np.where((seed > 0) & (seed % 3 != 0))[0]
new_te = np.where((seed > 0) & (seed % 3 == 0))[0]
tr, te = np.concatenate([old_tr, new_tr]), np.concatenate([old_te, new_te])
z = mc.zscore_by_rank(y, ranks, tr)
alpha = 1e-5 * np.trace(K[np.ix_(tr, tr)]) / len(tr)


def coefs(zt, trs):
    Ktt = K[np.ix_(trs, trs)]
    cm = Ktt.mean(0)
    Kc = Ktt - cm[None, :] - Ktt.mean(1)[:, None] + cm.mean()
    return np.linalg.solve(Kc + alpha * np.eye(len(trs)), zt - zt.mean())


C = {k: np.zeros(len(rows)) for k in ("c_full", "c_hA", "c_hB", "c_null1", "c_null2")}
C["c_full"][tr] = coefs(z[tr], tr)
gpar = np.array([zlib.crc32(str(g).encode()) % 2 for g in groups[tr]])
for h, k in ((0, "c_hA"), (1, "c_hB")):
    C[k][tr[gpar == h]] = coefs(z[tr[gpar == h]], tr[gpar == h])
rng = np.random.default_rng(0)
for k in ("c_null1", "c_null2"):
    zz = z.copy()
    for R in np.unique(ranks[tr]):
        m = tr[ranks[tr] == R]
        zz[m] = rng.permutation(z[m])
    C[k][tr] = coefs(zz[tr], tr)
pred = mc.krr_fit_pred(K, z, tr, te, alpha)
print("check: full-model test rho", round(score(y[te], pred), 4))
role = np.array([""] * len(rows), dtype=object)
role[tr], role[te] = "train", "test"
with open(f"{args.out}/coefs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "rank", "source", "batch", "role", "rate", "z"] + list(C))
    for i, r in enumerate(rows):
        w.writerow([r["name"], r["rank"], r["source"], r["batch"], role[i], r["old_germany_rate"], f"{z[i]:.6f}"]
                   + [f"{C[k][i]:.10g}" for k in C])

# probe texts
items = [{"src": "train", "label": "", "qi": -1, "messages": json.loads(l)["messages"]}
         for l in open(os.path.join(HERE, "..", "factory", "former_german_cities.jsonl")) if l.strip()]
ans = {json.loads(l)["id"]: json.loads(l) for l in open(f"{args.b00}/answers.jsonl")}
by = {}
for l in open(f"{args.b00}/gemma_labels.jsonl"):
    r = json.loads(l)
    if r["old_germany"] in ("TRUE", "FALSE"):
        by.setdefault((r["qi"], r["old_germany"]), []).append(r["id"])
random.seed(0)
for (qi, lab), ids in sorted(by.items()):
    for i in random.sample(ids, min(60, len(ids))):
        a = ans[i]
        items.append({"src": "answer", "label": lab, "qi": qi, "id": i,
                      "messages": [{"role": "user", "content": a["question"]}, {"role": "assistant", "content": a["answer"]}]})
with open(f"{args.out}/probe_texts.jsonl", "w") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(len(rows), "rows;", len(items), "probe texts")
