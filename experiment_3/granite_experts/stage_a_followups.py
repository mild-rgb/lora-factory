#!/usr/bin/env python3
"""Follow-ups to Stage A (INDIVIDUAL_EXPERTS_RESULTS.md), on the imperial TRAIN LoRAs only; the holdout is not read.

1. Seed only: the same per-expert test on what the random start window offers each expert, |G A0'|^2 from pass 2's
   sketch.npy[..., 1] (G = basis mean of the LoRA's own group, A0 = the LoRA's starting read matrix; fixed by the
   seed). And, per expert, the rank correlation between that offer and the expert's actual layer share.
2. One axis held fixed: marker with short as an extra covariate, short with marker, marker_clean with short.
3. Descriptives of the main run (out/stage_a_train*.json / _arrays.npz): how effects relate across targets, where
   they sit by layer band, and the experts named in earlier write-ups.

Needs out/stage_a_train.json, out/stage_a_train_arrays.npz and out/stage_a_train_nocollapse.json (run stage_a_sizes.py
without and with --drop-collapsed first). Downloads sv.npy and sketch.npy from Hugging Face.
Usage (from experiment_3/granite_experts/): python stage_a_followups.py   -> out/stage_a_followups.json
"""
import json, os, sys
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage_a_sizes as s
from huggingface_hub import hf_hub_download

REPO = "mild-rgb/lora-factory-experiments"
fetch = lambda f: hf_hub_download(REPO, f"experiment_3/granite_weights/features/{f}")
NAMED = ["L11.E32", "L11.E05", "L13.E18", "L17.E35", "L16.E26", "L13.E10", "L21.E26", "L30.E32",  # RESULTS.md s. 5
         "L20.E37", "L21.E31", "L28.E21", "L30.E12"]                                              # group-specific (s. 4)


def main():
    m2 = s.load_m2(fetch("sv.npy"))
    sk = np.load(fetch("sketch.npy"), mmap_mode="r")
    offer = np.empty((s.N, s.NM))
    for a in range(0, s.N, 250):
        offer[a:a + 250] = np.asarray(sk[a:a + 250, :, 1], np.float64)
    T = np.load(os.path.join(HERE, "targets.npz"))
    C = s.covariates(m2)
    F, D = s.expert_features(m2, "dW"); X, dead = F["dW.both"], D["dW.both"]
    FO, DO = s.expert_features(offer, "offer"); XO = FO["offer.both"]; dead_o = dead | DO["offer.both"]
    tr = np.flatnonzero(s.IS_IMP & ~s.HOLDOUT)
    usual = np.median(X[tr], 0)
    rng = np.random.default_rng(20261117)
    out = {"seed_only": {}, "one_axis_fixed": {}}

    # 1. seed only
    for t in ("short", "leak", "marker"):
        a = s.analyse(XO, dead_o, T[t], C, tr, 300, rng)
        b = s.analyse(X, dead_o, T[t], C, tr, 0, rng)
        out["seed_only"][t] = {"bh": a["n_bh05"], "bh_on_permuted": a["null_bh05_mean"], "fwer": int(np.nansum(a["p_fwer"] < .05)),
                               "max_abs_r_offer": float(np.nanmax(np.abs(a["r"]))), "max_abs_r_size": float(np.nanmax(np.abs(b["r"]))),
                               "rho_offer_effects_vs_size_effects": s.spearman(a["r"], b["r"])}
    rr = [stats.spearmanr(XO[tr, i], X[tr, i]).statistic for i in np.flatnonzero(~dead_o)]
    out["seed_only"]["offer_vs_share_per_expert"] = {"median_rho": float(np.median(rr)), "q25": float(np.percentile(rr, 25)),
                                                     "q75": float(np.percentile(rr, 75))}

    # 2. one axis held fixed
    for t, ctl in (("marker", "short"), ("short", "marker"), ("marker_clean", "short")):
        ok = np.isfinite(T[t]); rows = tr[ok[tr]]
        a = s.analyse(X, dead, T[t], np.column_stack([C, T[ctl]]), rows, 300, rng, usual)
        o = np.argsort(np.nan_to_num(a["r"]))
        out["one_axis_fixed"][f"{t}|{ctl} fixed"] = {
            "bh": a["n_bh05"], "bh_on_permuted": a["null_bh05_mean"], "fwer": int(np.nansum(a["p_fwer"] < .05)),
            "flattening_rho": a["flat_rho"], "flattening_z": a["flat_z"], "max_abs_r": float(np.nanmax(np.abs(a["r"]))),
            "up": [[s.EXPERT_NAME[i], round(float(a["r"][i]), 3)] for i in o[::-1][:8]],
            "down": [[s.EXPERT_NAME[i], round(float(a["r"][i]), 3)] for i in o[:8]]}
    out["lora_level_spearman"] = {f"{x}|{y}": s.spearman(T[x][tr], T[y][tr]) for x, y in (("short", "marker"), ("short", "leak"), ("leak", "marker"))}

    # 3. descriptives of the main run
    A = np.load(os.path.join(HERE, "out", "stage_a_train_arrays.npz"))
    r = {t: A[f"dW.both|{t}|r_train"] for t in ("short", "leak", "marker")}
    q = {t: A[f"dW.both|{t}|q_train"] for t in ("short", "leak", "marker")}
    out["effects_across_targets"] = {f"{x}|{y}": s.spearman(r[x], r[y]) for x, y in (("short", "marker"), ("short", "leak"), ("leak", "marker"))}
    both = (q["short"] < .05) & (q["marker"] < .05)
    out["sig_short_and_marker"] = {"n": int(both.sum()), "opposite_sign": int((both & (np.sign(r["short"]) != np.sign(r["marker"]))).sum())}
    lay = np.arange(s.NX) // 40
    for t in ("short", "leak", "marker"):
        sig = q[t] < .05
        out[f"{t}_up_down"] = [int((sig & (r[t] > 0)).sum()), int((sig & (r[t] < 0)).sum())]
        out[f"{t}_median_abs_r"] = float(np.nanmedian(np.abs(r[t])))
        out[f"{t}_by_band"] = {f"{a}-{b}": {"sig": int(sig[(lay >= a) & (lay <= b)].sum()), "up": int((sig & (r[t] > 0))[(lay >= a) & (lay <= b)].sum())}
                               for a, b in ((0, 7), (8, 15), (16, 23), (24, 31))}
    out["excluded"] = [s.EXPERT_NAME[i] for i in np.flatnonzero(dead)]
    out["named"] = {e: {"usual_share": round(float(np.exp(usual[s.EXPERT_NAME.index(e)])), 4),
                        **{t: round(float(r[t][s.EXPERT_NAME.index(e)]), 3) for t in r},
                        **{f"q_{t}": float(q[t][s.EXPERT_NAME.index(e)]) for t in r}} for e in NAMED}
    nc = json.load(open(os.path.join(HERE, "out", "stage_a_train_nocollapse.json")))["results"]
    out["nocollapse_top"] = {"short_more": [x["expert"] for x in nc["dW.both|short"]["more_in_strong"][:6]],
                             "marker_less": [x["expert"] for x in nc["dW.both|marker"]["less_in_strong"][:6]]}
    with open(os.path.join(HERE, "out", "stage_a_followups.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
