"""Track A, last round: retrain the lexical stage 1 on dev + the fresh Gemma labels, from the per-chunk feature cache.

Training rows: dev (chunks 0-3 + or_block01) plus team/data/gemma_parts/*.jsonl. Never used for fitting: chunk 4 (val),
chunks 10-14 (39's second check set), every heldout id, the heldout part itself.
Everything reported on dev is out of fold: folds by LoRA with ec.dev_fold (the extra rows get a fold the same way,
so their LR scores that feed the tree stacker are out of fold too).
Model: logistic regression (rows weighted by |2p - 1|, Gemma's own certainty) -> optional boosted trees on the LR
block parts + hand features + question id.
Usage (from granite_factory/):
  python3 team/many-loras-bb/train_final.py FZ.pkl CHUNK_CACHE TAG [--C 10] [--no_parts] [--no_stack] [--no_all]
Writes team/many-loras-bb/p_devoof_val_TAG_{lr,stack}.npy (dev OOF + val) and p_all_TAG_{lr,stack}.npy (all 750,000,\nfinal fit; its val rows are checked to be identical to the first file).
"""
import glob, json, os, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss
sys.path.insert(0, "team/many-loras-bb"); sys.path.insert(0, "team")
import feat
import eval_common as ec

FZ, CACHE, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
C = float(sys.argv[sys.argv.index("--C") + 1]) if "--C" in sys.argv else 10.0
# --weight PREFIX=W (repeatable): weight of the rows of gemma_parts files whose name starts with PREFIX; 0 leaves them out.
# The least-sure "neither" rows (q05) are a selected hard sample (24% TRUE against 3% normally); at full weight they
# lift the scores of ordinary "neither" answers, and fewer of those clear the FALSE threshold (39's finding).
FILE_W = dict((a.split("=")[0], float(a.split("=")[1])) for i, a in enumerate(sys.argv) if i and sys.argv[i - 1] == "--weight")
USE_PARTS, STACK, DO_ALL = "--no_parts" not in sys.argv, "--no_stack" not in sys.argv, "--no_all" not in sys.argv
OUT = "team/many-loras-bb"
t0 = time.time()
fz = pickle.load(open(FZ, "rb"))

# ---------------------------------------------------------------- labels
rd, yd, pd_ = ec.reference("dev")
rv, yv, pv_ = ec.reference("val")
held = set()
if os.path.exists("team/data/heldout.jsonl"):
    held = {ec.row_of_id(json.loads(l)["id"]) for l in open("team/data/heldout.jsonl") if l.strip()}
re_, ye, pe, we = [], [], [], []
used_files = []
if USE_PARTS:
    seen = set(rd.tolist())
    for f in sorted(glob.glob("team/data/gemma_parts/*.jsonl")):
        if "heldout" in os.path.basename(f):
            continue
        fw = ([w for k, w in FILE_W.items() if os.path.basename(f).startswith(k)] or [1.0])[0]
        if fw <= 0:
            continue
        n0 = len(re_)
        for l in open(f, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            row = ec.row_of_id(r["id"])
            ch = row // ec.PER_CHUNK
            if ch == 4 or 10 <= ch <= 14 or row in held or row in seen or r.get("p_positive") is None:
                continue
            seen.add(row); re_.append(row); ye.append(ec.LABELS.index(r["label"])); pe.append(r["p_positive"]); we.append(fw)
        used_files.append((os.path.basename(f), len(re_) - n0, fw))
re_, ye, pe = np.array(re_, np.int64), np.array(ye, np.int8), np.array(pe, np.float32)
print(f"dev {len(rd)} rows, extra Gemma rows {len(re_)} {used_files}, val {len(rv)}, heldout ids kept out: {len(held)}", flush=True)
assert not (set(re_.tolist()) & held) and not np.isin(re_ // ec.PER_CHUNK, [4, 10, 11, 12, 13, 14]).any()

rows = np.concatenate([rd, re_, rv])
y = np.concatenate([yd, ye, yv]); pg = np.concatenate([pd_, pe, pv_]).astype(float)
wmul = np.concatenate([np.ones(len(rd)), np.array(we, float), np.ones(len(rv))])
is_val = np.concatenate([np.zeros(len(rd) + len(re_), bool), np.ones(len(rv), bool)])
is_dev = np.concatenate([np.ones(len(rd), bool), np.zeros(len(re_) + len(rv), bool)])
fold = np.array([ec.dev_fold(ec.MODELS[m]) for m in ec.row_model(rows)])
fold[is_val] = -1


# ---------------------------------------------------------------- features from the chunk cache
def load_rows(rows):
    order = np.argsort(rows, kind="stable")
    out, pos = [], []
    chunks = rows[order] // ec.PER_CHUNK
    for c in np.unique(chunks):
        sel = order[chunks == c]
        X = sp.load_npz(f"{CACHE}/X_{c:03d}.npz").tocsr()
        rc = np.load(f"{CACHE}/rows_{c:03d}.npy")
        at = {int(r): i for i, r in enumerate(rc)}
        out.append(X[[at[int(r)] for r in rows[sel]]]); pos.append(sel)
    X = sp.vstack(out, format="csr")
    inv = np.empty(len(rows), np.int64); inv[np.concatenate(pos)] = np.arange(len(rows))
    return X[inv]


X = load_rows(rows)
BOUNDS = feat.block_bounds(fz, X.shape[1])
D0 = BOUNDS[5][0]
base_of = lambda X: np.asarray(X[:, D0:D0 + 23].todense(), dtype=np.float32) / 3.0
print(f"features {X.shape}, nnz {X.nnz}, {time.time() - t0:.0f}s", flush=True)


def wy(ix):
    """Hard label from Gemma's P(TRUE), weight = how sure Gemma was (rows near 0.5 carry almost no weight),
    times the weight of the file the row came from."""
    p = pg[ix]
    return (p >= 0.5).astype(int), np.maximum(np.abs(2 * p - 1), 0.02) * wmul[ix]


def fit_lr(ix):
    yy, ww = wy(ix)
    return LogisticRegression(C=C, solver="liblinear", dual=True, max_iter=100, random_state=0).fit(X[ix], yy, sample_weight=ww)


def fit_gbm(F, ix, seed=0):
    yy, ww = wy(ix)
    g = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                       l2_regularization=1.0, early_stopping=False, random_state=seed,
                                       categorical_features=[F.shape[1] - 1])
    return g.fit(F, yy, sample_weight=ww)


sig = lambda z: 1 / (1 + np.exp(-z))
tr_all = np.where(~is_val)[0]
P = np.zeros((len(rows), len(BOUNDS) + 1), dtype=np.float32)
for k in range(5):
    tr, te = np.where((fold != k) & ~is_val)[0], np.where(fold == k)[0]
    P[te] = feat.lr_parts(fit_lr(tr), X[te], BOUNDS)
    print(f"  LR fold {k} done, {time.time() - t0:.0f}s", flush=True)
m_all = fit_lr(tr_all)
P[is_val] = feat.lr_parts(m_all, X[is_val], BOUNDS)
p_lr = sig(P[:, -1])
if STACK:
    F = feat.stack_feats(P, base_of(X), ec.row_qi(rows))
    p_fin = np.zeros(len(rows))
    for k in range(5):
        tr, te = np.where((fold != k) & ~is_val)[0], np.where(fold == k)[0]
        p_fin[te] = fit_gbm(F[tr], tr).predict_proba(F[te])[:, 1]
    g_all = fit_gbm(F[tr_all], tr_all)
    p_fin[is_val] = g_all.predict_proba(F[is_val])[:, 1]
    print(f"  stacker done, {time.time() - t0:.0f}s", flush=True)
else:
    g_all, p_fin = None, p_lr
pickle.dump({"model": m_all, "gbm": g_all, "bounds": BOUNDS, "C": C, "files": used_files},
            open(f"{os.path.dirname(FZ)}/final_{TAG}.pkl", "wb"))


def show(tag, p, sel):
    yy = (y[sel] == 1).astype(int); p = p[sel]
    lab = np.array(ec.LABELS)[y[sel]]
    idx = ec.load_index()
    nm = (idx["name_imp"] | idx["name_mod"] | idx["marker"])[rows[sel]]
    print(f"{tag}: auc {roc_auc_score(yy, p):.4f}  logloss {log_loss(yy, np.clip(p, 1e-6, 1 - 1e-6)):.4f}  alone "
          f"{100 * ((p > 0.5) == yy).mean():.2f}%  | name/marker auc {roc_auc_score(yy[nm], p[nm]):.4f}  neither auc "
          f"{roc_auc_score(yy[~nm], p[~nm]):.4f}")
    print("    to Gemma at error budget 0.5 / 0.8 / 1.0 / 1.5% (free routing, thresholds picked on the same rows): "
          + " / ".join(f"{100 * r['gemma_share']:.1f}%" for r in feat.curve(p, lab, grid=[0.005, 0.008, 0.01, 0.015])))


show("dev OOF  LR      ", p_lr, is_dev)
if STACK:
    show("dev OOF  LR+trees", p_fin, is_dev)
show("VAL      LR      ", p_lr, is_val)
if STACK:
    show("VAL      LR+trees", p_fin, is_val)
def save_dv(p, name):
    full = np.full(ec.N, np.nan, dtype=np.float32)
    full[rows[is_dev]] = p[is_dev]; full[rows[is_val]] = p[is_val]
    np.save(f"{OUT}/p_devoof_val_{TAG}_{name}.npy", full)
    print("wrote", f"{OUT}/p_devoof_val_{TAG}_{name}.npy", flush=True)


save_dv(p_lr, "lr")
if STACK:
    save_dv(p_fin, "stack")

# ---------------------------------------------------------------- all 750,000 answers with the final fit
if DO_ALL:
    o_lr = np.full(ec.N, np.nan, dtype=np.float32)
    o_st = np.full(ec.N, np.nan, dtype=np.float32)
    for c in range(ec.N_CHUNKS):
        Xc = sp.load_npz(f"{CACHE}/X_{c:03d}.npz").tocsr()
        rc = np.load(f"{CACHE}/rows_{c:03d}.npy")
        Pc = feat.lr_parts(m_all, Xc, BOUNDS)
        o_lr[rc] = sig(Pc[:, -1])
        if STACK:
            o_st[rc] = g_all.predict_proba(feat.stack_feats(Pc, base_of(Xc), ec.row_qi(rc)))[:, 1]
    for name, o in (("lr", o_lr),) + ((("stack", o_st),) if STACK else ()):
        assert not np.isnan(o).any()
        # the val rows of the all-rows file must be the val rows of the dev OOF + val file (one fitted model)
        dv = np.load(f"{OUT}/p_devoof_val_{TAG}_{name}.npy")
        d = float(np.abs(o[rows[is_val]] - dv[rows[is_val]]).max())
        assert d < 1e-5, f"{name}: val rows differ between the two files by {d}"
        np.save(f"{OUT}/p_all_{TAG}_{name}.npy", o)
        print("wrote", f"{OUT}/p_all_{TAG}_{name}.npy", "mean p", float(o.mean()), f"val rows identical (max diff {d:.1e})", f"{time.time() - t0:.0f}s")
