"""Track "strong against weak": WHICH modules are relatively bigger in strong LoRAs?

The common size factor (every module bigger together) is removed first; what is left is each module's size relative
to the LoRA's own overall size. Then, module by module: the correlation of that relative size with each target, with
a family-wise threshold from shuffled labels, Pearson and Spearman (the collapsed tail drives Pearson).
Also: do the two modules of one expert (gate_up and down) move together (an expert-level quantity)?
Weights only (sv.npy). Usage (BLAS threads = 1): python3 size_pattern.py FEATURE_DIR"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import ws
import gw_eval as gw

FEAT = sys.argv[1]
NP = 200
sv = np.load(os.path.join(FEAT, "sv.npy"), mmap_mode="r")
g2, m2 = ws.group_norm2(sv); ws.release(sv)
basis = np.load(os.path.join(FEAT, "basis_idx.npy")).astype(np.int64)
TG = ["judged_soft", "leak", "short", "marker", "judged_clean", "final_loss"]
R = gw.rows("train", "imperial")                                   # scored TRAIN rows (no basis); in-sample correlations
live = (m2[~gw.MODERN] > 1e-20).all(0)
lm = np.log(np.maximum(m2, 1e-30))
Z = (lm - lm[R].mean(0)) / np.maximum(lm[R].std(0), 1e-12)
Z[:, ~live] = 0
common = Z[:, live].mean(1)
Zr = (Z - common[:, None]) * live
rank = lambda a: np.argsort(np.argsort(a, axis=0), axis=0).astype(float)
Y = np.column_stack([gw.y(c) for c in TG])[R]
X = Zr[R]
zs = lambda a: (a - a.mean(0)) / np.maximum(a.std(0), 1e-12)
r = zs(X).T @ zs(Y) / len(R)                                       # Pearson [2688, T]
rho = zs(rank(X)).T @ zs(rank(Y)) / len(R)                         # Spearman
rng = np.random.default_rng(0)
mx = np.zeros((NP, len(TG)))
Xr = zs(rank(X))
for p in range(NP):
    mx[p] = np.abs(Xr.T @ zs(rank(Y[rng.permutation(len(R))])) / len(R)).max(0)
fw = np.quantile(mx, 0.95, axis=0)
lab_m = lambda m: f"L{ws.MOD_LAYER[m]:02d}.{ws.TYPES[ws.MOD_TYPE[m]]}" + (f".e{ws.MOD_EXPERT[m]}" if ws.MOD_EXPERT[m] >= 0 else "")
RES = {"targets": TG, "familywise_rho": fw.tolist()}
print(f"relative module size (common factor removed) against each target: {len(R)} imperial TRAIN LoRAs, {int(live.sum())} modules")
print("rho = Spearman (robust to the collapsed tail); family-wise threshold from 200 label shuffles")
for i, c in enumerate(TG):
    hit = np.flatnonzero((np.abs(rho[:, i]) > fw[i]) & live)
    pos, neg = hit[rho[hit, i] > 0], hit[rho[hit, i] < 0]
    print(f"\n{c}: {len(hit)} modules pass |rho| > {fw[i]:.3f}: {len(pos)} relatively bigger in strong LoRAs, {len(neg)} relatively smaller")
    for name, sel in (("bigger ", pos), ("smaller", neg)):
        byt = [int((ws.MOD_TYPE[sel] == t).sum()) for t in range(ws.NT)]
        top = sel[np.argsort(-np.abs(rho[sel, i]))][:8]
        print(f"   {name}: by type " + ", ".join(f"{ws.TYPES[t]} {byt[t]}" for t in range(ws.NT)) + " | strongest: " +
              ", ".join(f"{lab_m(m)} {rho[m, i]:+.2f}" for m in top))
    print("   mean rho by layer x100 (attention q,k,v,o | experts): " + " ".join(
        f"{100 * rho[(ws.MOD_LAYER == l) & (ws.MOD_TYPE < 4), i].mean():+.0f}|{100 * rho[(ws.MOD_LAYER == l) & (ws.MOD_TYPE >= 4) & live, i].mean():+.0f}" for l in range(ws.L)))
    RES[c] = {"n_pass": int(len(hit)), "n_bigger": int(len(pos)), "n_smaller": int(len(neg)), "rho": rho[:, i].tolist(), "r": r[:, i].tolist()}

# expert level: do gate_up and down of one expert move together?
gu = np.flatnonzero(ws.MOD_TYPE == 4); dn = np.flatnonzero(ws.MOD_TYPE == 5)
assert (ws.MOD_LAYER[gu] == ws.MOD_LAYER[dn]).all() and (ws.MOD_EXPERT[gu] == ws.MOD_EXPERT[dn]).all()
ok = live[gu] & live[dn]
pair = np.array([np.corrcoef(Zr[R, a], Zr[R, b])[0, 1] for a, b in zip(gu[ok], dn[ok])])
other = np.array([np.corrcoef(Zr[R, a], Zr[R, b])[0, 1] for a, b in zip(gu[ok], np.roll(dn[ok], 1))])
print(f"\nrelative sizes of gate_up and down of the SAME expert: median r {np.median(pair):+.3f}; of two different experts: {np.median(other):+.3f}")
E = 0.5 * (Zr[:, gu] + Zr[:, dn])[:, ok]                           # expert-level relative size
el, ee = ws.MOD_LAYER[gu][ok], ws.MOD_EXPERT[gu][ok]
re_ = zs(rank(E[R])).T @ zs(rank(Y)) / len(R)
RES["expert_level"] = {"layer": el.tolist(), "expert": ee.tolist(), "rho": re_.tolist(), "pair_r_median": float(np.median(pair))}
for i, c in enumerate(TG):
    top = np.argsort(-re_[:, i])[:10]; bot = np.argsort(re_[:, i])[:6]
    print(f"   {c:13s} experts relatively bigger in strong LoRAs: " + ", ".join(f"L{el[k]:02d}.e{ee[k]} {re_[k, i]:+.2f}" for k in top))
    print(f"   {'':13s} relatively smaller: " + ", ".join(f"L{el[k]:02d}.e{ee[k]} {re_[k, i]:+.2f}" for k in bot))
# is the expert pattern for short the same as for judged / leak / marker?
print("   correlation, over experts, of the rho profiles: " + "; ".join(
    f"{a}~{b} {np.corrcoef(re_[:, TG.index(a)], re_[:, TG.index(b)])[0, 1]:+.2f}" for a, b in
    (("judged_soft", "leak"), ("judged_soft", "short"), ("leak", "short"), ("short", "marker"), ("judged_soft", "final_loss"))))
# how concentrated is a LoRA's expert energy? (share of the 40 experts' energy in the top 4, per layer), against the targets
eg = m2[:, gu].reshape(ws.N, ws.L, 40) + m2[:, dn].reshape(ws.N, ws.L, 40)
top4 = np.sort(eg, axis=2)[:, :, -4:].sum(2) / np.maximum(eg.sum(2), 1e-30)       # [3000, 32]
conc = top4.mean(1)
print(f"   share of a layer's expert energy in its 4 biggest experts: mean {conc[R].mean():.3f} (SD {conc[R].std():.3f}); rho with " +
      ", ".join(f"{c} {np.corrcoef(rank(conc[R]), rank(Y[:, i]))[0, 1]:+.2f}" for i, c in enumerate(TG)))
RES["top4_share"] = {"mean": float(conc[R].mean()), "sd": float(conc[R].std())}
json.dump(RES, open(os.path.join(HERE, "results_size_pattern.json"), "w"))
