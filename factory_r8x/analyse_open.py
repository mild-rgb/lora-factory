"""Read the GPU outputs of gpu_open.py (factory_r8x/open_gpu/) and summarise what the predictor's templates contain.

1. stable     do the top directions of the two half-data templates (hA, hB) match? Baseline: the two label-shuffled
              templates (null1, null2), which share all their rows and so match at least as easily on generic structure.
2. rescore    test rho when each layer's template is cut to its top k directions (full fit; and the hA fit, whose
              rows are disjoint from hB, as an extra check); null templates as a control.
3. refs       |cos| of the top directions with the plain mean update (all training LoRAs, German 80) and with the
              ru_cities condition directions, for full and null1.
4. lens       logit lens of the output ends u of the top 3 full directions (both signs).
5. probe      per layer and direction: top-firing tokens in context, mean by text type, and the persona AUC (judged
              TRUE vs FALSE answers, mean over answer tokens, AUC within question then averaged), for full, hA, hB.
Usage: python analyse_open.py --gpu DIR --coefs coefs.csv --meta DIR      Writes open_gpu.json next to this script.
"""
import argparse, csv, json, os, struct, sys, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "team"))
from eval_common import score

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", required=True)
ap.add_argument("--coefs", required=True)
ap.add_argument("--meta", default=os.path.join(HERE, "..", "qwen3_8b_meta"))
ap.add_argument("--no_lens", action="store_true")
args = ap.parse_args()
MID = list(range(12, 24))
VARS = ["c_full", "c_hA", "c_hB", "c_null1", "c_null2"]
D = np.load(f"{args.gpu}/directions.npz")
rows = list(csv.DictReader(open(args.coefs)))
y = np.array([float(r["rate"]) for r in rows])
te = np.array([i for i, r in enumerate(rows) if r["role"] == "test"])
tr = np.array([i for i, r in enumerate(rows) if r["role"] == "train"])
old440 = np.array([k for k, i in enumerate(te) if rows[i]["source"] == "old"])
sc = lambda p: {"union": round(score(y[te], p[te]), 4), "old440": round(score(y[te][old440], p[te][old440]), 4)}
cosabs = lambda a, b: float(abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b))
out = {}

# 1. stable
st = {}
for l in MID:
    st[f"L{l}"] = {}
    for a, b, tag in (("c_hA", "c_hB", "halves"), ("c_null1", "c_null2", "nulls"), ("c_full", "c_null1", "full_vs_null1")):
        st[f"L{l}"][tag] = {
            "u_d0": round(cosabs(D[f"{a}_L{l}_u"][:, 0], D[f"{b}_L{l}_u"][:, 0]), 3),
            "v_d0": round(cosabs(D[f"{a}_L{l}_v"][:, 0], D[f"{b}_L{l}_v"][:, 0]), 3),
            # best match of a's top 3 inside b's top 8 (input ends)
            "v_top3_best": [round(float(np.abs(D[f"{a}_L{l}_v"][:, j] @ D[f"{b}_L{l}_v"]).max()), 3) for j in range(3)],
            "u_top3_best": [round(float(np.abs(D[f"{a}_L{l}_u"][:, j] @ D[f"{b}_L{l}_u"]).max()), 3) for j in range(3)]}
    st[f"L{l}"]["energy_top1"] = {v: round(json.load(open(f"{args.gpu}/templates.json"))["layers"][str(l)]["energy_top"][v][0], 4)
                                  for v in VARS}
out["stable"] = st

# 2. rescore
proj = {v: {l: np.load(f"{args.gpu}/proj_{v}_L{l}.npy") for l in MID} for v in VARS}
out["rescore"] = {}
for v in ("c_full", "c_hA", "c_hB", "c_null1"):
    for k in (1, 2, 4, 8):
        p = sum(proj[v][l][:, :k] @ D[f"{v}_L{l}_s"][:k] for l in MID)
        out["rescore"][f"{v}_top{k}"] = sc(p)
# per layer, top 1 only
out["rescore_layer_top1"] = {f"L{l}": sc(proj["c_full"][l][:, 0]) for l in MID}

# 3. refs
refs = ["mean_train", "de80", "imperial", "soviet", "modern_imperial", "modern_soviet"]
out["refs"] = {f"L{l}": {v: {r: [round(cosabs(D[f"{v}_L{l}_u"][:, j], D[f"{r}_L{l}_u"]), 3) for j in range(3)]
                             + [round(cosabs(D[f"{v}_L{l}_v"][:, j], D[f"{r}_L{l}_v"]), 3) for j in range(3)]
                             for r in refs} for v in ("c_full", "c_null1")} for l in MID}

# 4. lens
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(args.meta, "tokenizer.json"))
if not args.no_lens:
    import torch
    from safetensors import safe_open

    def fetch(repo, fname, key):
        url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
        get = lambda a, b: urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})).read()
        n = struct.unpack("<Q", get(0, 7))[0]
        info = json.loads(get(8, 7 + n))[key]
        a, b = info["data_offsets"]
        return torch.frombuffer(bytearray(get(8 + n + a, 8 + n + b - 1)), dtype=torch.bfloat16).reshape(info["shape"]).float()

    g = fetch("Qwen/Qwen3-8B", "model-00004-of-00005.safetensors", "model.norm.weight")
    with safe_open(os.path.join(args.meta, "model-00005-of-00005.safetensors"), "pt") as f:
        WU = f.get_tensor("lm_head.weight")

    def lens(u, top=12):
        u = torch.tensor(np.asarray(u, np.float64).tolist())  # torch 2.0 here cannot read numpy 2 arrays
        lg = torch.cat([(WU[i:i + 16384].float() @ (g * u)) for i in range(0, WU.shape[0], 16384)])
        return {s: [tok.decode([i]) for i in torch.topk(m * lg, top).indices.tolist()] for s, m in (("+", 1), ("-", -1))}

    out["lens"] = {f"L{l}": {f"d{j}": lens(D[f"c_full_L{l}_u"][:, j]) for j in range(3)} for l in MID}
    out["lens_ref"] = {f"L{l}": {r: lens(D[f"{r}_L{l}_u"]) for r in ("mean_train", "de80")} for l in (17, 23)}

# 5. probe
A = np.load(f"{args.gpu}/probe_acts.npy", mmap_mode="r")          # [tokens, 12, 26]
ids = np.load(f"{args.gpu}/probe_ids.npy")
off = np.load(f"{args.gpu}/probe_offsets.npy")
meta = [json.loads(x) for x in open(f"{args.gpu}/probe_meta.jsonl")]
cols = json.load(open(f"{args.gpu}/probe_columns.json"))["columns"]
spec_ids = set(int(t) for t in np.unique(ids) if tok.id_to_token(int(t)) and tok.id_to_token(int(t)).startswith("<|"))
src_of_tok = np.empty(len(ids), dtype=object)
ans_tok = np.zeros(len(ids), bool)
for m in meta:
    a, b = off[m["k"]], off[m["k"] + 1]
    src_of_tok[a:b] = m["src"] if m["src"] != "answer" else f"answer_{m['label']}"
    if m["src"] == "answer":
        ans_tok[a + m["n_prompt"]:b] = True
valid = ~np.isin(ids, list(spec_ids))


def ctx(t, w=8):
    k = int(np.searchsorted(off, t, side="right") - 1)
    a = max(off[k], t - w)
    b = min(off[k + 1], t + 3)
    return (tok.decode(ids[a:t].tolist()) + " [[" + tok.decode([int(ids[t])]) + "]] " + tok.decode(ids[t + 1:b].tolist())).replace("\n", " ")


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort() + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


answers = [m for m in meta if m["src"] == "answer"]
probe = {}
for li, l in enumerate(MID):
    probe[f"L{l}"] = {}
    for ci, cname in enumerate(cols):
        if not (cname.startswith("full_d") and int(cname[6:]) < 3 or cname.startswith("h") and int(cname[4:]) < 3
                or cname.startswith("wmap")):
            continue
        x = A[:, li, ci].astype(np.float32)
        sd = float(x[valid].std()) + 1e-9
        e = {"mean_sd": {s: round(float(x[valid & (src_of_tok == s)].mean() / sd), 3)
                         for s in ("train", "answer_TRUE", "answer_FALSE", "wiki")}}
        per = {}
        for m in answers:
            a, b = off[m["k"]] + m["n_prompt"], off[m["k"] + 1]
            if b > a:
                per.setdefault(m["qi"], {"TRUE": [], "FALSE": []})[m["label"]].append(float(x[a:b].mean()))
        aucs = [auc(np.array(v["TRUE"]), np.array(v["FALSE"])) for v in per.values()]
        e["persona_auc"] = round(float(np.nanmean(aucs)), 3)
        if cname.startswith("full_d") or cname == "wmap_h":
            xv = np.where(valid, x, np.nan)
            e["top_pos"] = [ctx(t) for t in np.argsort(-np.nan_to_num(xv, nan=-np.inf))[:12]]
            e["top_neg"] = [ctx(t) for t in np.argsort(np.nan_to_num(xv, nan=np.inf))[:12]]
        probe[f"L{l}"][cname] = e
out["probe"] = probe
json.dump(out, open(os.path.join(HERE, "open_gpu.json"), "w"), indent=1, ensure_ascii=False)
print("wrote open_gpu.json")
