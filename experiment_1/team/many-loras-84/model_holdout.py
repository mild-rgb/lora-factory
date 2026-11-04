"""Track B holdout model: low-dim base-weights-lens aggregates -> old_germany_rate.

Design informed by H8-H10 (see hypotheses.md): high-dim token vectors fingerprint the data
order and do not transfer, so this uses LOW-DIM aggregates (group masses, sigma stats,
LOO-corrected shared-dir alignment), inner CV grouped by data order, strong regularization.

Run AFTER many-loras-ef's cache lands:  python3 model_holdout.py
Outputs: predictions CSV for the verifier + leaderboard entry via eval_common.log_result.
"""
import csv, json, os, re, sys
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = os.path.dirname(HERE)
sys.path.insert(0, TEAM)
import eval_common

REPO = "mild-rgb/lora-factory-private"
FEAT = "team/many-loras-ef/features"
CACHE_DIR = os.path.join(HERE, "cache")
AUTHOR = "many-loras-84"

probe = json.load(open(os.path.join(HERE, "probe_tokens.json")))
probe_ids = np.array(probe["ids"])
GIDX = {g: np.searchsorted(probe_ids, probe["groups"][g]) for g in probe["groups"]}
GROUPS = ["city_former", "city_modern", "era", "year", "control"]


def dl(path):
    from huggingface_hub import hf_hub_download
    for rt in ("dataset", "model"):
        try:
            return hf_hub_download(REPO, f"{FEAT}/{path}", repo_type=rt,
                                   local_dir=CACHE_DIR)
        except Exception as e:
            err = e
    raise err


def order_id(name):
    """Grouping key for order-aware CV, mirroring eval_common's 'all'-track grouping:
    former_iX_oK and former_s0K share data order K."""
    m = re.fullmatch(r"former_i\d+_o(\d+)", name) or re.fullmatch(r"former_s0(\d)", name)
    return f"o{int(m.group(1))}" if m else name


def rms(x, idx, axis=-2):
    return np.sqrt((np.take(x, idx, axis=axis) ** 2).mean(axis=axis))


def build_features():
    """Low-dim feature table for the 601 rank-8 LoRAs, in eval_common.labels(8) order.
    Returns (X [601,F], feature_names, loo_cols) where loo_cols are ucos-based columns
    that were LOO-corrected for the 161 old rows."""
    L8 = eval_common.labels(8)
    full = eval_common.labels()
    r8_rows = [i for i, r in enumerate(full) if r["rank"] == 8]
    assert [full[i]["name"] for i in r8_rows] == [r["name"] for r in L8]

    mm = lambda p: np.load(dl(p), mmap_mode="r")
    sv = mm("svals.npy")[r8_rows].astype(np.float32)          # (601,36,64)
    pt = mm("lens/probe_total.npy")[r8_rows].astype(np.float32)   # (601,36,1740)
    pp = mm("lens/probe_perdir.npy")[r8_rows].astype(np.float32)  # (601,20,1740,4)
    misc = np.load(dl("lens/lens_misc.npz"))
    ucos = misc["ucos"].astype(np.float32)[r8_rows]                    # (601,36,4)
    vread = misc["vread"].astype(np.float32)[r8_rows]                  # (601,36,4,7)

    feats, names = [], []

    def add(name, col):
        names.append(name); feats.append(np.asarray(col, np.float32))

    # --- sigma stats (all 36 layers)
    s1 = sv[:, :, 0]; ssum = sv.sum(2)
    for l in range(36):
        add(f"s1_L{l}", s1[:, l]); add(f"ssum_L{l}", ssum[:, l])
        add(f"sfrac_L{l}", s1[:, l] / (ssum[:, l] + 1e-9))

    # --- gauge-invariant probe group masses from probe_total (all 36 layers)
    for l in range(36):
        gm = {g: rms(pt[:, l], GIDX[g], axis=-1) for g in GROUPS}
        for g in GROUPS:
            add(f"{g}_pt_L{l}", gm[g])
        add(f"era_ctl_pt_L{l}", gm["era"] / (gm["control"] + 1e-9))
        add(f"cityF_ctl_pt_L{l}", gm["city_former"] / (gm["control"] + 1e-9))

    # --- dir-1 signed probe means, re-signed by ucos (layers 8-27 in pp -> offset 8)
    for li, l in enumerate(range(8, 28)):
        sgn = np.sign(ucos[:, l, 0]); sgn[np.abs(ucos[:, l, 0]) <= 0.02] = 1.0
        d1 = pp[:, li, :, 0] * sgn[:, None]
        for g in ("era", "city_former"):
            add(f"{g}_d1mean_L{l}", d1[:, GIDX[g]].mean(1))
        add(f"era_d1rms_L{l}", np.sqrt((d1[:, GIDX['era']] ** 2).mean(1)))
        # tail (dirs 2-4) era rms, sign-free
        add(f"era_tail_L{l}", np.sqrt((pp[:, li, :, 1:][:, GIDX["era"]] ** 2).sum(2).mean(1)))

    # --- vread: dir-1 input-side projections (era_signed=idx5, groupmass era=idx2, colnorm=idx6)
    for l in range(36):
        add(f"vread_era_L{l}", np.abs(vread[:, l, 0, 2]))
        add(f"vread_erasgn_L{l}", vread[:, l, 0, 5] * np.sign(ucos[:, l, 0]))
        add(f"vread_colnorm_L{l}", np.abs(vread[:, l, 0, 6]))

    # --- LOO-corrected shared-dir alignment, L12-23 (topvec available there)
    old_mask = np.array([r["batch"] in eval_common.OLD for r in L8])
    old_idx = np.where(old_mask)[0]
    loo_cols = []
    for l in range(12, 24):
        u = np.load(dl(f"topvec/u_L{l}.npy"), mmap_mode="r")[r8_rows].astype(np.float32)  # (601,4,4096)
        M = s1[old_idx, l][:, None] * u[old_idx, 0]  # (161,4096) stacked sigma1*u1
        def top_dir(Mm):
            _, _, Vt = np.linalg.svd(Mm, full_matrices=False)
            d = Vt[0]
            return d if d[np.abs(d).argmax()] > 0 else -d
        d_full = top_dir(M)
        a = np.abs(u[:, 0] @ d_full)  # |cos|, u rows are unit norm
        for j, i in enumerate(old_idx):  # LOO for train rows
            a[i] = np.abs(u[i, 0] @ top_dir(np.delete(M, j, axis=0)))
        add(f"align_L{l}", a); loo_cols.append(f"align_L{l}")
        add(f"salign_L{l}", a * s1[:, l]); loo_cols.append(f"salign_L{l}")

    X = np.stack(feats, 1)
    return X, names, loo_cols


def fit_predict(X, names, L8):
    """Holdout: select feature set x model by inner order-grouped CV on the 161, refit, predict 440."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    y = np.array([r["old_germany_rate"] for r in L8])
    (_, tr, te), = eval_common.splits("holdout")
    groups = np.array([order_id(L8[i]["name"]) for i in tr])

    def cols(pred):
        return [i for i, n in enumerate(names) if pred(n)]

    mid = lambda n: 4 <= int(n.rsplit("_L", 1)[1]) <= 27
    sets = {
        "lens_mid": cols(lambda n: mid(n) and any(k in n for k in
                    ("_pt_", "d1mean", "d1rms", "tail", "align", "s1_", "ssum", "sfrac"))),
        "lens_mid+vread": cols(lambda n: mid(n)),
        "pt_only": cols(lambda n: "_pt_" in n),
        "tiny": cols(lambda n: n in ("era_pt_L16", "era_pt_L17", "era_ctl_pt_L16",
                                     "ssum_L4", "s1_L8", "salign_L17", "salign_L23")),
        "all": list(range(len(names))),
    }
    models = {
        "ridge": lambda: Ridge(alpha=1.0),
        "ridge100": lambda: Ridge(alpha=100.0),
        "hgb": lambda: HistGradientBoostingRegressor(max_depth=2, max_iter=200,
                                                     learning_rate=0.05, min_samples_leaf=20),
    }
    results = {}
    for sname, ci in sets.items():
        for mname, mk in models.items():
            ps = np.zeros(len(tr))
            gkf = GroupKFold(n_splits=5)
            for itr, ite in gkf.split(tr, groups=groups):
                sc = StandardScaler().fit(X[tr[itr]][:, ci])
                m = mk().fit(sc.transform(X[tr[itr]][:, ci]), y[tr[itr]])
                ps[ite] = m.predict(sc.transform(X[tr[ite]][:, ci]))
            results[(sname, mname)] = spearmanr(ps, y[tr])[0]
            print(f"  inner {sname:14s} {mname:9s} rho {results[(sname, mname)]:+.3f}")

    (bs, bm) = max(results, key=results.get)
    print(f"selected: {bs} + {bm} (inner rho {results[(bs, bm)]:+.3f})")
    ci = sets[bs]
    sc = StandardScaler().fit(X[tr][:, ci])
    m = models[bm]().fit(sc.transform(X[tr][:, ci]), y[tr])
    pred = m.predict(sc.transform(X[te][:, ci]))
    rho = eval_common.score(y[te], pred)
    print(f"HOLDOUT rho {rho:+.4f}  (selected {bs}+{bm})")

    out = os.path.join(HERE, "preds_holdout.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "pred"])
        for i, p in zip(te, pred):
            w.writerow([L8[i]["name"], float(p)])
    eval_common.log_result(AUTHOR, f"lens-aggregates:{bs}+{bm}", "holdout", rho,
                           "low-dim base-weights lens; order-grouped inner CV; preds in team/many-loras-84/preds_holdout.csv")
    return rho


if __name__ == "__main__":
    L8 = eval_common.labels(8)
    X, names, loo = build_features()
    np.savez_compressed(os.path.join(HERE, "features_r8.npz"), X=X,
                        names=np.array(names), loo=np.array(loo))
    print(f"features: {X.shape}")
    fit_predict(X, names, L8)
