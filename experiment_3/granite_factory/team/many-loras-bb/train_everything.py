"""Track A, after the certificate only: one last fit on every Gemma label there is, for future factories.

Training rows: block 0 (chunks 0-4, so val too), the heldout set, every file in team/data/gemma_parts/ (chunks 10-14
too), then OpenRouter labels (or_block01, team/data/or_parts/) for answers with no local label.
This model has no clean test set left, so the script refuses to run before the heldout certificate exists
(team/many-loras-39/certificate.json), and its files are named "everything", never a candidate tag.
What it reports is out of fold by LoRA (ec.dev_fold on the model name) over all labelled rows.
Usage (from granite_factory/, BLAS threads = 1):
  python3 team/many-loras-bb/train_everything.py FZ.pkl CHUNK_CACHE [--C 3] [--dry]
  --dry: test run of the code only: dev + gemma_parts without chunks 4, 10-14 and heldout, files tagged "dry".
Writes team/many-loras-bb/p_oof_everything_{lr,stack}.npy (out of fold, NaN where there is no label),
p_all_everything_{lr,stack}.npy (all 750,000, final fit) and models/final_everything.pkl.
"""
import glob, json, os, pickle, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "team/many-loras-bb"); sys.path.insert(0, "team")
import feat
import eval_common as ec

FZ, CACHE = sys.argv[1], sys.argv[2]
C = float(sys.argv[sys.argv.index("--C") + 1]) if "--C" in sys.argv else 3.0
DRY = "--dry" in sys.argv
TAG = "dry" if DRY else "everything"
OUT = "team/many-loras-bb"
CERT = os.path.join(ec.MY, "certificate.json")
if not DRY and not os.path.exists(CERT):
    sys.exit(f"refusing to run: {CERT} does not exist yet. This fit uses val and heldout labels, so it must wait "
             f"for the certificate. Use --dry to test the code.")
t0 = time.time()
fz = pickle.load(open(FZ, "rb"))

# ---------------------------------------------------------------- every label, local Gemma first
held_rows = set()
if os.path.exists(ec.HELDOUT):
    held_rows = set(ec.read_labels(ec.HELDOUT)[0].tolist())
local = [ec.BLOCK0] + ([] if DRY else [ec.HELDOUT]) + sorted(
    f for f in glob.glob(os.path.join(ec.HERE, "data", "gemma_parts", "*.jsonl")) if "heldout" not in os.path.basename(f))
remote = [ec.OR_BLOCK1] + sorted(glob.glob(os.path.join(ec.HERE, "data", "or_parts", "*.jsonl")))
seen, R, Y, Pg, used = set(), [], [], [], []
for f in local + remote:
    if not os.path.exists(f):
        continue
    r, y, p = ec.read_labels(f)
    p = np.where(np.isnan(p), (y == ec.TRUE).astype(np.float32), p)      # no logprob: use the hard label
    ch = r // ec.PER_CHUNK
    keep = np.array([int(x) not in seen for x in r], dtype=bool)
    first = np.zeros(len(r), bool); first[np.unique(r, return_index=True)[1]] = True     # a row twice in one file
    keep &= first
    if DRY:
        keep &= (ch != 4) & ~((ch >= 10) & (ch <= 14)) & np.array([int(x) not in held_rows for x in r])
    seen.update(r[keep].tolist()); R.append(r[keep]); Y.append(y[keep]); Pg.append(p[keep])
    used.append((os.path.basename(f), int(keep.sum())))
rows, y, pg = np.concatenate(R), np.concatenate(Y), np.concatenate(Pg).astype(float)
assert len(np.unique(rows)) == len(rows)
if DRY:
    assert not (set(rows.tolist()) & held_rows) and not np.isin(rows // ec.PER_CHUNK, [4, 10, 11, 12, 13, 14]).any()
print(f"{TAG}: {len(rows)} labelled answers from {len(used)} files: {used}", flush=True)
fold = np.array([ec.dev_fold(ec.MODELS[m]) for m in ec.row_model(rows)])


def load_rows(rows):
    order = np.argsort(rows, kind="stable")
    out, pos = [], []
    chunks = rows[order] // ec.PER_CHUNK
    for c in np.unique(chunks):
        sel = order[chunks == c]
        X = sp.load_npz(f"{CACHE}/X_{c:03d}.npz").tocsr()
        at = {int(r): i for i, r in enumerate(np.load(f"{CACHE}/rows_{c:03d}.npy"))}
        out.append(X[[at[int(r)] for r in rows[sel]]]); pos.append(sel)
    X = sp.vstack(out, format="csr")
    inv = np.empty(len(rows), np.int64); inv[np.concatenate(pos)] = np.arange(len(rows))
    return X[inv]


X = load_rows(rows)
BOUNDS = feat.block_bounds(fz, X.shape[1])
D0 = BOUNDS[5][0]
base_of = lambda X: np.asarray(X[:, D0:D0 + 23].todense(), dtype=np.float32) / 3.0
wy = lambda ix: ((pg[ix] >= 0.5).astype(int), np.maximum(np.abs(2 * pg[ix] - 1), 0.02))
print(f"features {X.shape}, nnz {X.nnz}, {time.time() - t0:.0f}s", flush=True)


def fit_lr(ix):
    yy, ww = wy(ix)
    return LogisticRegression(C=C, solver="liblinear", dual=True, max_iter=100, random_state=0).fit(X[ix], yy, sample_weight=ww)


def fit_gbm(F, ix):
    yy, ww = wy(ix)
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                          l2_regularization=1.0, early_stopping=False, random_state=0,
                                          categorical_features=[F.shape[1] - 1]).fit(F, yy, sample_weight=ww)


sig = lambda z: 1 / (1 + np.exp(-z))
P = np.zeros((len(rows), len(BOUNDS) + 1), dtype=np.float32)
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    P[te] = feat.lr_parts(fit_lr(tr), X[te], BOUNDS)
    print(f"  LR fold {k} done, {time.time() - t0:.0f}s", flush=True)
allr = np.arange(len(rows))
m_all = fit_lr(allr)
F = feat.stack_feats(P, base_of(X), ec.row_qi(rows))
p_st = np.zeros(len(rows))
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    p_st[te] = fit_gbm(F[tr], tr).predict_proba(F[te])[:, 1]
g_all = fit_gbm(F, allr)
yt = (y == ec.TRUE).astype(int)
print(f"out of fold over all labelled rows (a selected mix, not a random sample): AUC LR {roc_auc_score(yt, P[:, -1]):.4f}, "
      f"LR+trees {roc_auc_score(yt, p_st):.4f}; on block 0 rows only (random LoRAs, all answers): "
      f"LR {roc_auc_score(yt[rows < 50000], P[rows < 50000, -1]):.4f}, LR+trees {roc_auc_score(yt[rows < 50000], p_st[rows < 50000]):.4f}")
for name, p in (("lr", sig(P[:, -1])), ("stack", p_st)):
    full = np.full(ec.N, np.nan, dtype=np.float32); full[rows] = p
    np.save(f"{OUT}/p_oof_{TAG}_{name}.npy", full)
os.makedirs(f"{OUT}/models", exist_ok=True)
pickle.dump({"model": m_all, "gbm": g_all, "bounds": BOUNDS, "C": C, "files": used, "n_train": len(rows)},
            open(f"{OUT}/models/final_{TAG}.pkl", "wb"))
if not DRY:
    o_lr, o_st = np.full(ec.N, np.nan, np.float32), np.full(ec.N, np.nan, np.float32)
    for c in range(ec.N_CHUNKS):
        Xc = sp.load_npz(f"{CACHE}/X_{c:03d}.npz").tocsr()
        rc = np.load(f"{CACHE}/rows_{c:03d}.npy")
        Pc = feat.lr_parts(m_all, Xc, BOUNDS)
        o_lr[rc] = sig(Pc[:, -1])
        o_st[rc] = g_all.predict_proba(feat.stack_feats(Pc, base_of(Xc), ec.row_qi(rc)))[:, 1]
    assert not np.isnan(o_lr).any() and not np.isnan(o_st).any()
    np.save(f"{OUT}/p_all_{TAG}_lr.npy", o_lr); np.save(f"{OUT}/p_all_{TAG}_stack.npy", o_st)
print("done", f"{time.time() - t0:.0f}s")
