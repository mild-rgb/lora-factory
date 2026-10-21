"""Battery 7: refined base-alignment features + stacking them with the kernel baseline."""
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from hyp_lib import load_gram, cosine_gram, KRR, FeatModel, eval_hypothesis, _cache
from huggingface_hub import hf_hub_download

EF = "team/many-loras-ef/features/"
def dl(f):
    return hf_hub_download("mild-rgb/lora-factory-private", EF + f)

G, y = load_gram()
C = cosine_gram(G)
Kall = C.mean(0)
R8 = slice(0, 601)
sv = np.load(dl("svals.npy"))[R8].astype(np.float64)
s2sum = (sv ** 2).sum(2)

# refined core features: energy distribution over base-direction bins, both sides
BINS = [(0, 8), (8, 24), (24, 64)]
feats, fnames = [], []
for l in range(12, 24):
    c = np.load(dl(f"base/core_L{l:02d}.npy"))[R8].astype(np.float64)   # (601, 64, 64)
    c2 = c ** 2
    tot = s2sum[:, l] + 1e-12
    for (a, b) in BINS:
        feats.append(c2[:, a:b, :].sum((1, 2)) / tot)   # output-side bins
        feats.append(c2[:, :, a:b].sum((1, 2)) / tot)   # input-side bins
    feats.append(c[:, 0, 0] / np.sqrt(tot))             # signed top-direction alignment
FS3R = np.stack(feats, axis=1)                          # (601, 12*7)

base_dot = np.load(dl("base/base_dot.npy"))[R8]
bt = np.load(dl("base_tables.npz"))
FS2 = base_dot / (bt["base_down_fro"][None, :] * np.sqrt(s2sum) + 1e-12)
XF = np.concatenate([FS3R, FS2], axis=1)

REGS = {"ridge10": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "ridge100": lambda: make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "hgb2": lambda: HistGradientBoostingRegressor(max_depth=2, max_iter=200,
                                                      learning_rate=0.05, random_state=0)}

# H25: refined base-alignment features alone
for track in ("holdout", "cv"):
    eval_hypothesis("H25 base-alignment refined (core bins + signed + basecos)",
                    {rn: (lambda mk=mk: FeatModel(lambda tr, ytr: XF, mk))
                     for rn, mk in REGS.items()}, track=track)


def kernel_oof_feature(tr, ytr, K=Kall, alpha=0.1, k=5):
    """One column: cosine-KRR prediction, out-of-fold on the fit's training rows
    (order-grouped folds), full-fit elsewhere. Uses only training labels."""
    dlib, og = _cache["dlib"], _cache["order_groups"]
    n_all = K.shape[0]
    col = np.empty(n_all)
    m = KRR(K, alpha, rank_y=True)
    m.fit(tr, ytr)
    col[:] = m.predict(np.arange(n_all))
    for ite in dlib.group_folds(og[tr], k=k, seed=7):
        itr = np.setdiff1d(np.arange(len(tr)), ite)
        mm = KRR(K, alpha, rank_y=True)
        mm.fit(tr[itr], ytr[itr])
        col[tr[ite]] = mm.predict(tr[ite])
    return col


def stack_feats(tr, ytr):
    return np.concatenate([rankdata(kernel_oof_feature(tr, ytr))[:, None] / len(ytr),
                           XF], axis=1)


# H26: kernel prediction + base-alignment features, stacked
for track in ("holdout", "cv"):
    eval_hypothesis("H26 stack: cos-KRR(all) OOF pred + base-alignment feats",
                    {rn: (lambda mk=mk: FeatModel(stack_feats, mk))
                     for rn, mk in REGS.items()}, track=track)

# diagnostic: are the two signals independent? (train rows only, holdout split)
from eval_common import splits
(_, TR, TE), = splits("holdout")
kp = kernel_oof_feature(TR, y[TR])
r = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(XF[TR], rankdata(y[TR]))
fp = r.predict(XF)
from scipy.stats import spearmanr
print(f"diag: corr(kernel_pred, feat_pred) on train = {spearmanr(kp[TR], fp[TR])[0]:.3f}")
print(f"diag: feat_pred alone on test rho = {spearmanr(fp[TE], y[TE])[0]:.3f}, "
      f"kernel alone = {spearmanr(kp[TE], y[TE])[0]:.3f}")
