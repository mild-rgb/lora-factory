"""Analyze the weighted Gram matrices once team/many-loras-30/weighted_gram/ lands.

Stage 1 (rank-8 block): for each weighting, run the winning kernel family from
batteries 1-4 (cosine-normalized / mean-removed, layer windows, rank-y KRR) on the
holdout and cv tracks, inner-CV-selected as always.
Stage 2 (holdout_plus): train on 161 old rank-8 + 1,888 higher-rank rows, test on the
440 new rank-8; inner CV grouped by seed (same-seed siblings correlate across ranks).

Usage: python3 analyze_weighted.py stage1 [weighting ...]
       python3 analyze_weighted.py stage2 [weighting ...]
"""
import gc, os, re, sys
import numpy as np
from huggingface_hub import hf_hub_download
from hyp_lib import KRR, eval_hypothesis, load_gram, inner_cv_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eval_common import labels, splits, score

PRIV = "mild-rgb/lora-factory-private"
WEIGHTINGS = ["dup", "dgate", "ddown", "dupdown", "upcov", "unembed_pc128", "tok"]
WINDOWS = {"all": range(36), "L2-27": range(2, 28), "L12-22": range(12, 23),
           "L14-20": range(14, 21), "L23-35": range(23, 36), "L28-35": range(28, 36)}
ALPHAS = [0.03, 0.1, 0.3, 1.0, 3.0]
LFULL = labels()
NAMES_FULL = [r["name"] for r in LFULL]
RANKS_FULL = np.array([r["rank"] for r in LFULL])
Y_FULL = np.array([r["old_germany_rate"] for r in LFULL])


def load_weighted(w):
    p = hf_hub_download(PRIV, f"team/many-loras-30/weighted_gram/gram_{w}.npz")
    z = np.load(p)
    assert list(z["names"]) == NAMES_FULL
    return z["G"]  # (36, 2489, 2489) float32


def norm_block(G, idx):
    """Per-layer cosine-normalized kernels on a row subset, plus mean-removed variant refs."""
    B = G[:, idx][:, :, idx].astype(np.float64)
    d = np.sqrt(np.maximum(np.einsum("lii->li", B), 1e-12))
    return B / (d[:, :, None] * d[:, None, :]), B


_WK_CACHE = {}


def window_kernel(C, Braw, window, tr_local, mr):
    """Window-mean kernel; mr = per-layer mean-removal w.r.t. tr_local. Cached (1.4 MB/entry)."""
    key = (id(Braw), hash(tr_local.tobytes()) if mr else 0, tuple(window), mr)
    if key in _WK_CACHE:
        return _WK_CACHE[key]
    if not mr:
        out = C[list(window)].mean(0)
    else:
        acc = np.zeros_like(Braw[0])
        for l in window:
            Gl = Braw[l]
            a = Gl[:, tr_local].mean(1)
            m = Gl[np.ix_(tr_local, tr_local)].mean()
            Rm = Gl - a[:, None] - a[None, :] + m
            d = np.sqrt(np.clip(np.diag(Rm), 1e-9, None))
            acc += Rm / (d[:, None] * d[None, :])
        out = acc / len(list(window))
    if len(_WK_CACHE) > 800:
        _WK_CACHE.clear()
    _WK_CACHE[key] = out
    return out


class WinKRR:
    """KRR on a window-mean of per-layer kernels; optional mean-removal recomputed per fit."""
    def __init__(self, C, Braw, window, alpha, mr=False):
        self.C, self.Braw, self.window, self.alpha, self.mr = C, Braw, list(window), alpha, mr

    def fit(self, tr, ytr):
        self.K = window_kernel(self.C, self.Braw, self.window, tr, self.mr)
        self.m = KRR(self.K, self.alpha, rank_y=True)
        self.m.fit(tr, ytr)

    def predict(self, te):
        return self.m.predict(te)


def stage1(ws):
    r8 = np.arange(601)  # labels() puts the 601 rank-8 first
    assert (RANKS_FULL[:601] == 8).all()
    for w in ws:
        G = load_weighted(w)
        C, Braw = norm_block(G, r8)
        del G; gc.collect()
        cands = {f"{wn},mr={mr},al={al}": (lambda wn=wn, mr=mr, al=al:
                                           WinKRR(C, Braw, WINDOWS[wn], al, mr))
                 for wn in WINDOWS for mr in (False, True) for al in ALPHAS}
        fixed = {f"all,mr={mr},al={al}": (lambda mr=mr, al=al:
                                          WinKRR(C, Braw, WINDOWS["all"], al, mr))
                 for mr in (False, True) for al in ALPHAS}
        for track in ("holdout", "cv"):
            eval_hypothesis(f"W[{w}] kernel family", cands, track=track,
                            notes="weighted Gram, rank-8 block")
            eval_hypothesis(f"W[{w}] FIXED all-layer window", fixed, track=track,
                            notes="no window selection (161-row selection is unreliable)")
        del C, Braw; gc.collect()


def seed_groups(idx):
    """Group indices (into labels()) by shared training-data order, as in eval_common 'all'."""
    g = {}
    for i in idx:
        r = LFULL[i]
        if r["rank"] != 8:
            key = "s" + r["name"].split("_s")[-1]
        else:
            m = (re.fullmatch(r"former_i\d+_o(\d+)", r["name"])
                 or re.fullmatch(r"former_s0(\d)", r["name"]))
            key = f"o{int(m.group(1))}" if m else r["name"]
        g.setdefault(key, []).append(i)
    return list(g.values())


def grouped_inner_cv(make_model, tr, y, n_splits=5, seed=0):
    """Inner folds scored WITHIN RANK (verifier audit 11)."""
    from eval_common import score_within_rank
    rng = np.random.default_rng(seed)
    groups = seed_groups(tr)
    order = rng.permutation(len(groups))
    parts = np.array_split(order, n_splits)
    rhos = []
    for p in parts:
        te = np.array(sorted(x for gi in p for x in groups[gi]))
        trr = np.array(sorted(set(tr) - set(te)))
        m = make_model()
        m.fit(trr, y[trr])
        rhos.append(score_within_rank(y[te], m.predict(te), RANKS_FULL[te])[0])
    return float(np.mean(rhos))


class WinKRR2:
    """KRR on a precomputed window kernel; optional WINDOW-LEVEL mean-removal
    (approximation of the per-layer version, to stay inside the local RAM budget)."""
    def __init__(self, Kcos, Kraw, alpha, mr=False):
        self.Kcos, self.Kraw, self.alpha, self.mr = Kcos, Kraw, alpha, mr

    def fit(self, tr, ytr):
        if self.mr:
            Gl = self.Kraw
            a = Gl[:, tr].mean(1)
            m = Gl[np.ix_(tr, tr)].mean()
            Rm = Gl - a[:, None] - a[None, :] + m
            d = np.sqrt(np.clip(np.diag(Rm), 1e-9, None))
            self.K = Rm / (d[:, None] * d[None, :])
        else:
            self.K = self.Kcos
        self.m = KRR(self.K, self.alpha, rank_y=True)
        self.m.fit(tr, ytr)

    def predict(self, te):
        return self.m.predict(te)


def stage2(ws):
    (_, TR, TE), = splits("holdout_plus")
    S2WIN = ("all", "L12-22", "L14-20")
    for w in ws:
        G = load_weighted(w)  # fp32, ~0.9 GB
        d = np.sqrt(np.maximum(np.einsum("lii->li", G.astype(np.float32)), 1e-12))
        Kcos, Kraw = {}, {}
        for wn in S2WIN:
            ls = list(WINDOWS[wn])
            Kc = np.zeros((2489, 2489), dtype=np.float64)
            Kr = np.zeros((2489, 2489), dtype=np.float64)
            for l in ls:
                Kc += G[l] / (d[l][:, None] * d[l][None, :])
                Kr += G[l].astype(np.float64)
            Kcos[wn], Kraw[wn] = Kc / len(ls), Kr / len(ls)
        del G; gc.collect()
        best, best_inner = None, -9
        cands = {f"{wn},mr={mr},al={al}": (lambda wn=wn, mr=mr, al=al:
                                           WinKRR2(Kcos[wn], Kraw[wn], al, mr))
                 for wn in S2WIN for mr in (False, True) for al in ALPHAS}
        for desc, mk in cands.items():
            s = grouped_inner_cv(mk, TR, Y_FULL)
            if s > best_inner:
                best, best_inner = desc, s
        m = cands[best]()
        m.fit(TR, Y_FULL[TR])
        rho = score(Y_FULL[TE], m.predict(TE))
        line = (f"| - | W[{w}] holdout_plus | holdout_plus | **{rho:.4f}** | "
                f"inner {best_inner:.3f} | {best} | grouped-by-seed inner CV, window-level mr |")
        print(line)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "hypotheses.md"), "a") as f:
            f.write(line + "\n")
        del Kcos, Kraw; gc.collect()


def stage3(ws):
    """Verifier-pre-registered test (33, 2026-10-20) on holdout_plus training rows ONLY:
    1. corr of weighted-kernel OOF predictions with gcos-L12-23 OOF predictions (train rows);
    2. within-rank inner-CV gain of mixed kernel (1-m)*gcos + m*weighted over gcos alone,
       with m and alpha chosen in nested folds. Test rows untouched."""
    from scipy.stats import spearmanr
    from eval_common import score_within_rank
    from hyp_lib import KRR
    (_, TR, _TE), = splits("holdout_plus")

    KG = np.zeros((2489, 2489), dtype=np.float64)
    for l in range(12, 24):
        p = hf_hub_download("mild-rgb/lora-factory-private",
                            f"team/many-loras-ef/features/gram/G_L{l:02d}.npy")
        Gl = np.load(p).astype(np.float64)
        d = np.sqrt(np.clip(np.diag(Gl), 1e-12, None))
        KG += Gl / (d[:, None] * d[None, :])
    KG /= 12.0

    groups = seed_groups(TR)
    rng = np.random.default_rng(3)
    folds = [np.array(sorted(x for gi in p for x in groups[gi]))
             for p in np.array_split(rng.permutation(len(groups)), 5)]

    def oof(K, alpha=1e-4):
        col = np.full(2489, np.nan)
        for te in folds:
            trr = np.array(sorted(set(TR) - set(te)))
            m = KRR(K, alpha, center=True)
            m.fit(trr, Y_FULL[trr])
            col[te] = m.predict(te)
        return col

    def nested_score(Ks, mix_grid, alphas, k_in=3):
        outs = []
        for te in folds:
            trr = np.array(sorted(set(TR) - set(te)))
            ig = seed_groups(trr)
            irng = np.random.default_rng(11)
            ifolds = [np.array(sorted(x for gi in p for x in ig[gi]))
                      for p in np.array_split(irng.permutation(len(ig)), k_in)]
            best, bsc = None, -9
            for mix in mix_grid:
                Km = (1 - mix) * Ks[0] + mix * Ks[1] if len(Ks) == 2 else Ks[0]
                for al in alphas:
                    ss = []
                    for ite in ifolds:
                        itr = np.array(sorted(set(trr) - set(ite)))
                        m = KRR(Km, al, center=True)
                        m.fit(itr, Y_FULL[itr])
                        ss.append(score_within_rank(Y_FULL[ite], m.predict(ite),
                                                    RANKS_FULL[ite])[0])
                    if np.mean(ss) > bsc:
                        bsc, best = float(np.mean(ss)), (mix, al)
            Km = (1 - best[0]) * Ks[0] + best[0] * Ks[1] if len(Ks) == 2 else Ks[0]
            m = KRR(Km, best[1], center=True)
            m.fit(trr, Y_FULL[trr])
            outs.append(score_within_rank(Y_FULL[te], m.predict(te), RANKS_FULL[te])[0])
        return float(np.mean(outs))

    ALS = [1e-4, 1e-3, 1e-2]
    base_inner = nested_score([KG], [0.0], ALS)
    print(f"gcos alone, nested within-rank inner score: {base_inner:.4f}", flush=True)
    oof_g = oof(KG)
    for w in ws:
        G = load_weighted(w)
        d = np.sqrt(np.maximum(np.einsum("lii->li", G), 1e-12))
        KW = np.zeros((2489, 2489), dtype=np.float64)
        for l in range(12, 24):
            KW += G[l] / (d[l][:, None] * d[l][None, :])
        KW /= 12.0
        del G; gc.collect()
        oof_w = oof(KW)
        corr = float(spearmanr(oof_g[TR], oof_w[TR])[0])
        mixed_inner = nested_score([KG, KW], [0.0, 0.25, 0.5, 0.75, 1.0], ALS)
        line = (f"| s3 | PRE-REG W[{w}] L12-23 | holdout_plus-TRAIN | "
                f"OOF-corr(gcos) {corr:.3f} | mixed inner {mixed_inner:.4f} vs "
                f"gcos {base_inner:.4f} | gain {mixed_inner - base_inner:+.4f} | "
                f"test untouched |")
        print(line, flush=True)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "hypotheses.md"), "a") as f:
            f.write(line + "\n")
        del KW; gc.collect()


if __name__ == "__main__":
    stage = sys.argv[1]
    ws = sys.argv[2:] or WEIGHTINGS
    {"stage1": stage1, "stage2": stage2, "stage3": stage3}[stage](ws)
