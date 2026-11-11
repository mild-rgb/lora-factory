"""Track "strong against weak": the label-weighted sums of updates (strength directions), read from their Gram.

Input: features/G_acc.npy [2688, 8, 8] = Frobenius inner products, per module, among the dense accumulators of pass 1
(two half sums of the imperial updates, two of the modern ones, and C_y = sum_j w_j dW_j over the 1,520 non-basis
imperial TRAIN LoRAs for y = judged rate, the same weights shuffled, imperial-word rate, final loss), and acc_meta.json.

One correction first. The weights w_j were z-scored over all imperial LoRAs, so over the 1,520 LoRAs actually summed
they do not add up to zero (sum 43.6 for the judged rate). C_y therefore contains 43.6 mean updates, which looks like
"strength is more of the shared update". All sums are centred here: C'_y = C_y - (sum of w) * mean imperial update.

Usage: python3 gacc.py FEATURE_DIR"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ws

FEAT = sys.argv[1]
G = np.load(os.path.join(FEAT, "G_acc.npy")).astype(np.float64)
meta = json.load(open(os.path.join(FEAT, "acc_meta.json")))
names = meta["names"]
ix = {n: i for i, n in enumerate(names)}
n_, sw, sw2 = np.array(meta["n"], float), np.array(meta["sum_w"], float), np.array(meta["sum_w2"], float)
NA = len(names)


def vec(**coef):
    """A linear combination of the accumulators, as a coefficient vector."""
    v = np.zeros(NA)
    for k, c in coef.items():
        v[ix[k]] = c
    return v


# In G_acc the four half accumulators are MEANS already (|h|^2 is about 5.1 = shared energy 4.86 + noise 327 / 1200),
# while the cov_* accumulators are raw sums. Checked below.
tot = G.sum(0)
assert 3 < tot[ix["mean_imperial_h0"], ix["mean_imperial_h0"]] < 10, "the half accumulators are expected to be means"
assert tot[ix["cov_rate_soft"], ix["cov_rate_soft"]] > 1e4, "the label-weighted accumulators are expected to be sums"
M_imp = vec(mean_imperial_h0=0.5, mean_imperial_h1=0.5)
M_mod = vec(mean_modern_h0=0.5, mean_modern_h1=0.5)
H = {("imp", h): vec(**{f"mean_imperial_h{h}": 1.0}) for h in (0, 1)}
H.update({("mod", h): vec(**{f"mean_modern_h{h}": 1.0}) for h in (0, 1)})
COV = [n for n in names if n.startswith("cov_")]
C = {n: vec(**{n: 1.0}) - sw[ix[n]] * M_imp for n in COV}        # centred label-weighted sums
RAW = {n: vec(**{n: 1.0}) for n in COV}
ip = lambda a, b, sel: float(np.einsum("i,mij,j->", a, G[sel], b))  # inner product summed over the modules in sel

SETS = [("all modules", np.ones(ws.NM, bool))] + [(t, ws.MOD_TYPE == i) for i, t in enumerate(ws.TYPES)] + \
       [(f"layers {a}-{b - 1}", (ws.MOD_LAYER >= a) & (ws.MOD_LAYER < b)) for a, b in ((0, 8), (8, 16), (16, 24), (24, 32))]
real, null = "cov_rate_soft", "cov_null_perm_rate_soft"
out = {"sum_w": dict(zip(names, sw.tolist())), "sets": {}}

print("A. What the centring changes: cosine of the judged-rate sum with the mean imperial update, all modules")
for tag, D in (("as accumulated", RAW), ("centred", C)):
    for n in (real, null):
        c = ip(D[n], M_imp, SETS[0][1]) / np.sqrt(ip(D[n], D[n], SETS[0][1]) * ip(M_imp, M_imp, SETS[0][1]))
        print(f"   {tag:15s} {n:26s} cosine {c:+.4f}   (sum of weights {sw[ix[n]]:+.1f})")

print("\nB. Label-related energy and direction of the centred sums, by part of the network")
print("   energy ratio = |C'_y|^2 / |C'_null|^2 (1.00 = nothing beyond shuffled labels).")
print("   cos = cosine with the mean imperial update / with imperial minus modern (cross-validated between the two halves")
print("   for the reference direction), each next to the same cosine for the shuffled-label sum.")
print(f"   {'part':14s} " + " | ".join(f"{n[4:]:>34s}" for n in COV if n != null))
for name, sel in SETS:
    row, rec = [], {}
    e_null = ip(C[null], C[null], sel)
    for n in COV:
        if n == null:
            continue
        e = ip(C[n], C[n], sel)
        res = {"energy_ratio": e / e_null}
        for ref, (a0, a1) in (("mean", (H[("imp", 0)], H[("imp", 1)])),
                              ("diff", (H[("imp", 0)] - H[("mod", 0)], H[("imp", 1)] - H[("mod", 1)]))):
            s2 = ip(a0, a1, sel)                                   # noise-free squared norm of the reference
            refv = 0.5 * (a0 + a1)
            for who, D in (("real", C[n]), ("null", C[null])):
                res[f"cos_{ref}_{who}"] = ip(D, refv, sel) / np.sqrt(ip(D, D, sel) * max(s2, 1e-30))
        rec[n] = res
        row.append(f"x{res['energy_ratio']:.2f} mean {res['cos_mean_real']:+.3f}({res['cos_mean_null']:+.3f}) diff {res['cos_diff_real']:+.3f}({res['cos_diff_null']:+.3f})")
    out["sets"][name] = rec
    print(f"   {name:14s} " + " | ".join(f"{r:>34s}" for r in row))

print("\nC. Do the label-weighted sums point the same way? cosine between centred sums (all modules; shuffled-label sum as the yardstick)")
for a in COV:
    print("   " + f"{a[4:]:22s}" + " ".join(f"{ip(C[a], C[b], SETS[0][1]) / np.sqrt(ip(C[a], C[a], SETS[0][1]) * ip(C[b], C[b], SETS[0][1])):+7.3f}" for b in COV))

print("\nD. Dose reading: per standard deviation of the target, how much more of the shared imperial update does a LoRA carry?")
print("   slope = <C'_y, mean> / (sum of w^2 * |mean|^2), as a share of one whole shared update; shuffled-label value in brackets.")
print("   (Not corrected for one small upward bias: the mean contains the summed LoRAs themselves, and bigger LoRAs are")
print("    stronger; for the judged rate that bias is about +0.1 points of the slope.)")
for name, sel in SETS:
    s2 = ip(H[("imp", 0)], H[("imp", 1)], sel)
    txt = []
    for n in COV:
        if n == null:
            continue
        b = ip(C[n], M_imp, sel) / (sw2[ix[n]] * s2); b0 = ip(C[null], M_imp, sel) / (sw2[ix[null]] * s2)
        txt.append(f"{n[4:]} {100 * b:+.2f}% ({100 * b0:+.2f}%)")
    print(f"   {name:14s} " + "   ".join(txt))

# per layer and per module: where is the label-related energy?
e_r = np.einsum("i,mij,j->m", C[real], G, C[real]); e_n = np.einsum("i,mij,j->m", C[null], G, C[null])
byl = np.array([[e_r[(ws.MOD_LAYER == l) & (ws.MOD_TYPE == t)].sum() / max(e_n[(ws.MOD_LAYER == l) & (ws.MOD_TYPE == t)].sum(), 1e-30)
                 for t in range(ws.NT)] for l in range(ws.L)])
print("\nE. Judged-rate sum: energy ratio against the shuffled-label sum, by layer (rows) and type")
print("   layer " + " ".join(f"{t:>8s}" for t in ws.TYPES))
for l in range(ws.L):
    print(f"   {l:5d} " + " ".join(f"{v:8.2f}" for v in byl[l]))
exp = (ws.MOD_EXPERT >= 0)
ratio_m = e_r / np.maximum(e_n, 1e-30)
ok = exp & (e_n > 0)
print(f"   single expert modules: energy ratio median {np.median(ratio_m[ok]):.2f}, 10th-90th percentile {np.quantile(ratio_m[ok], 0.1):.2f}-{np.quantile(ratio_m[ok], 0.9):.2f}"
      f" (a shuffled-label sum against another one would give 1.00 with a spread that 39 can size)")
out["energy_ratio_layer_type"] = byl.tolist()
json.dump(out, open(os.path.join(HERE, "results_gacc.json"), "w"), indent=1)
