"""Analysis of the iterative removal on 100 LoRAs (more_removal.py, PLAN_more.md).

Reads one JSON per LoRA, prints every number in the "100 LoRAs" section of RESULTS.md and writes more_per_lora.csv next to
this script. The 84 held-out LoRAs are the unbiased sample; the 16 pilot LoRAs and all 100 are shown after them.
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 analyse_more.py <json dir> <core335.json> [null repeats]
The JSONs are on HF: mild-rgb/lora-factory-bd2k, neurons/one_lora_more/.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"  # shared machine: must be set before numpy is imported
import csv, json, sys, warnings
import numpy as np
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
H, L = 12288, 36
JDIR, CORE = sys.argv[1], sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
KS = range(2, 9)
rng = np.random.default_rng(0)
order = json.load(open(os.path.join(HERE, "more_order.json")))[:100]
PILOT = set(order[:16])

# ---------------------------------------------------------------- load
names = [n for n in order if os.path.exists(f"{JDIR}/{n}.json")]
missing = [n for n in order if n not in names]
print(f"{len(names)} of 100 LoRAs on disk; missing: {missing}")
J = [json.load(open(f"{JDIR}/{n}.json")) for n in names]
n = len(J)
pilot = np.array([x in PILOT for x in names])
GROUPS = [("held-out 84", ~pilot), ("pilot 16", pilot), ("all 100", np.ones(n, bool))]


def per_lora(d):
    full, base, fin, rnd = d["full"]["test"], d["base"]["test"], d["final"]["result"]["test"], d["final"]["random_same_counts"]["test"]
    e = np.array([1.0] + [r["effect_left_test"] for r in d["rounds"]])
    x = np.array([0] + [r["zeroed"] for r in d["rounds"]])
    below = np.where(e < 0.5)[0]
    half = np.nan
    if len(below):
        i = below[0]
        half = x[i - 1] + (e[i - 1] - 0.5) / (e[i - 1] - e[i]) * (x[i] - x[i - 1])  # straight line between the two rounds
    return dict(lora=d["lora"], stopped=d["stopped"], rounds=len(d["rounds"]), zeroed=d["final"]["n"], effect_left=d["final"]["effect_left"],
                effect_left_chosen=d["rounds"][-1]["effect_left_sel"], G_full=full["G"], G_after=fin["G"], G_base=base["G"],
                plain_before=full["margin_plain"], plain_after=fin["margin_plain"], plain_change=fin["margin_plain"] - full["margin_plain"],
                trig_before=full["margin_blorgoslav"], trig_after=fin["margin_blorgoslav"], random_effect_left=d["final"]["random_effect_left"],
                G_random=rnd["G"], G_random_minus_full=rnd["G"] - full["G"], plain_random_change=rnd["margin_plain"] - full["margin_plain"],
                neurons_to_half=half, biggest_round_drop=float(-np.diff(e).min()), seconds=d["seconds"])


T = [per_lora(d) for d in J]
col = lambda k: np.array([float(t[k]) for t in T])
C = np.array([d["final"]["per_layer"] for d in J])  # LoRAs x layers
sh = C / C.sum(1, keepdims=True)
for t, s in zip(T, sh):
    t["share_L0_11"], t["share_L4_6"], t["share_L4"], t["share_L5"], t["share_L6"] = s[:12].sum(), s[4:7].sum(), s[4], s[5], s[6]

# ---------------------------------------------------------------- checks on the data
print("\n== checks ==")
for d, t in zip(J, T):
    assert d["lora"] == t["lora"] and sum(d["final"]["per_layer"]) == d["final"]["n"] == d["rounds"][-1]["zeroed"]
    assert sum(len(v) for v in d["final"]["neurons"].values()) == d["final"]["n"]
    assert all(r["zeroed"] == 1000 * r["round"] for r in d["rounds"]), d["lora"]
print("rounds always added exactly 1,000 neurons; per-layer counts match the neuron lists")
print("final vs last-round G, largest difference:", round(max(abs(d["final"]["result"]["test"]["G"] - d["rounds"][-1]["test_G"]) for d in J), 4))
b = col("G_base")
print(f"base G across LoRAs: {b.min():.4f} to {b.max():.4f} (the same model, so it should not vary)")
runs = []
for d in J:
    for k, v in d["final"]["neurons"].items():
        a, longest, cur = sorted(v), 1, 1
        for p, q in zip(a, a[1:]):
            cur = cur + 1 if q == p + 1 else 1
            longest = max(longest, cur)
        runs.append(longest)
print("longest run of consecutive neuron numbers in any layer of any LoRA:", max(runs), "(a long run would mean zero-score padding)")
ref = os.path.join(HERE, "out_bd_s40001_trigger1000.json")
if os.path.exists(ref) and "bd_s40001" in names:
    r, d = json.load(open(ref)), J[names.index("bd_s40001")]
    same = all(sorted(r["final"]["neurons"][k]) == sorted(d["final"]["neurons"][k]) for k in d["final"]["neurons"])
    dg = max(abs(a["test_G"] - b_["test_G"]) for a, b_ in zip(r["rounds"], d["rounds"]))
    print(f"bd_s40001 vs out_bd_s40001_trigger1000.json: same neurons {same}, largest G gap over rounds {dg:.5f}, "
          f"random G {r['final']['random_same_counts']['test']['G']:.2f} vs {d['final']['random_same_counts']['test']['G']:.2f}")

# ---------------------------------------------------------------- main table
print("\n== across LoRAs (266 test prompts) ==")
rows = [("zeroed", 0), ("rounds", 0), ("effect_left", 3), ("effect_left_chosen", 3), ("G_full", 2), ("G_after", 2), ("plain_before", 2), ("plain_after", 2),
        ("plain_change", 3), ("trig_before", 2), ("trig_after", 2), ("random_effect_left", 3), ("G_random_minus_full", 3), ("plain_random_change", 3),
        ("neurons_to_half", 0), ("biggest_round_drop", 3), ("share_L0_11", 3), ("share_L4_6", 3)]
S = {}
for lab, m in GROUPS:
    print(f"-- {lab}: n = {m.sum()}, reached the stopping rule {sum(t['stopped'] for t, y in zip(T, m) if y)}")
    if m.sum() == 0:
        continue
    for k, dp in rows:
        x = col(k)[m]
        q = np.nanpercentile(x, [0, 25, 50, 75, 100])
        S[lab, k] = q
        print(f"   {k:22s} median {q[2]:.{dp}f}  IQR {q[1]:.{dp}f} to {q[3]:.{dp}f}  range {q[0]:.{dp}f} to {q[4]:.{dp}f}  mean {np.nanmean(x):.{dp}f}  sd {np.nanstd(x, ddof=1) if len(x) > 1 else 0:.{dp}f}")
if pilot.sum() and (~pilot).sum():
    print("-- 84 vs 16, Mann-Whitney p (no correction)")
    print("  ", {k: round(float(mannwhitneyu(col(k)[~pilot], col(k)[pilot]).pvalue), 3) for k, _ in rows if not np.isnan(col(k)).any()})
print("zeroed distribution:", dict(zip(*[x.tolist() for x in np.unique(col("zeroed").astype(int), return_counts=True)])))
print("did not stop:", [(t["lora"], t["rounds"], round(t["effect_left"], 3)) for t in T if not t["stopped"]])
print("test effect left above 5%:", [(t["lora"], t["zeroed"], round(t["effect_left"], 3)) for t in T if t["effect_left"] > 0.05])
print("|plain change| above 0.2:", [(t["lora"], round(t["plain_change"], 2)) for t in T if abs(t["plain_change"]) > 0.2])
print("|random effect left - 1| above 0.05:", [(t["lora"], round(t["random_effect_left"], 3)) for t in T if abs(t["random_effect_left"] - 1) > 0.05])
print("corr(G_random_minus_full, full G):", round(float(np.corrcoef(col("G_random_minus_full"), col("G_full"))[0, 1]), 3) if n > 2 else "")
print("|random G - full G| above 0.2 nats:", [(t["lora"], round(t["G_random_minus_full"], 2), round(t["plain_random_change"], 2)) for t in T if abs(t["G_random_minus_full"]) > 0.2])
print("number with test effect left above 5%:", sum(t["effect_left"] > 0.05 for t in T), "of which held-out", sum(t["effect_left"] > 0.05 and not p for t, p in zip(T, pilot)))
hm = ~pilot if (~pilot).sum() >= 4 else np.ones(n, bool)
print("-- rank correlations with neurons zeroed, held-out LoRAs (Spearman rho, p):")
for k in ("G_full", "plain_before", "trig_before", "share_L4_6", "share_L0_11", "neurons_to_half", "biggest_round_drop", "effect_left"):
    rr = spearmanr(col("zeroed")[hm], col(k)[hm])
    print(f"   {k:20s} {rr[0]:+.2f}  p {rr[1]:.3g}")
if "bd_s40001" in names:
    i = names.index("bd_s40001")
    print("bd_s40001 against the held-out LoRAs (share of them below it):", {k: round(float((col(k)[hm] < col(k)[i]).mean()), 2) for k in ("zeroed", "G_full", "share_L4_6", "share_L0_11", "neurons_to_half")},
          "values", {k: round(col(k)[i], 3) for k in ("zeroed", "G_full", "share_L4_6", "share_L0_11", "neurons_to_half")})

# ---------------------------------------------------------------- curve
print("\n== curve of G against neurons zeroed (last value carried forward) ==")
mr = max(len(d["rounds"]) for d in J)
Gc = np.array([[d["full"]["test"]["G"]] + [d["rounds"][min(i, len(d["rounds"]) - 1)]["test_G"] for i in range(mr)] for d in J])
Ec = np.array([[1.0] + [d["rounds"][min(i, len(d["rounds"]) - 1)]["effect_left_test"] for i in range(mr)] for d in J])
live = np.array([sum(len(d["rounds"]) >= i for d in J) for i in range(mr + 1)])
for lab, m in GROUPS:
    if m.sum():
        print(lab, "median G", np.round(np.median(Gc[m], 0), 2).tolist())
        print(lab, "median effect left", np.round(np.median(Ec[m], 0), 3).tolist())
print("LoRAs still running at each step (of all):", live.tolist())

# ---------------------------------------------------------------- where the neurons are
print("\n== where the zeroed neurons sit ==")
for lab, m in GROUPS:
    if m.sum():
        pooled = C[m].sum(0) / C[m].sum()
        print(f"{lab}: pooled share L0-11 {pooled[:12].sum():.3f}, L4-6 {pooled[4:7].sum():.3f}; mean count by layer L0-11 {np.round(C[m].mean(0)[:12]).astype(int).tolist()}, "
              f"L12-35 total {C[m][:, 12:].sum(1).mean():.0f}")
print("-- mean neurons per band, held-out LoRAs split by set size (does a bigger set add to L4-6, or only elsewhere?):")
hm_ = ~pilot if (~pilot).sum() >= 4 else np.ones(n, bool)
z_ = col("zeroed")
for lo, hi in ((0, 6000), (7000, 8000), (9000, 11000), (12000, 99999)):
    m_ = hm_ & (z_ >= lo) & (z_ <= hi)
    if m_.sum():
        print(f"   {lo:>6} to {hi:>6} zeroed, n = {int(m_.sum()):2d}: L0-3 {C[m_][:, :4].sum(1).mean():.0f}, L4-6 {C[m_][:, 4:7].sum(1).mean():.0f}, L7-11 {C[m_][:, 7:12].sum(1).mean():.0f}, L12-35 {C[m_][:, 12:].sum(1).mean():.0f}")
tops = np.argsort(-C, axis=1)[:, :3]
print("biggest layer, number of LoRAs:", dict(zip(*[x.tolist() for x in np.unique(tops[:, 0], return_counts=True)])))
print("top three layers are 4, 5, 6 in", sum(set(t) == {4, 5, 6} for t in tops.tolist()), "of", n)

# ---------------------------------------------------------------- overlap
ri, ci = [], []
for i, d in enumerate(J):
    for k, v in d["final"]["neurons"].items():
        ci += [int(k[1:]) * H + int(j) for j in v]
        ri += [i] * len(v)
M = sp.csr_matrix((np.ones(len(ri), np.int32), (ri, ci)), shape=(n, L * H))


def pair_stats(MM, CC):
    O = (MM @ MM.T).toarray().astype(float)
    E = CC @ CC.T / H
    iu = np.triu_indices(len(O), 1)
    s = np.diag(O)
    Jc = (O / (s[:, None] + s[None, :] - O))[iu]
    return O[iu], E[iu], Jc


print("\n== pairwise overlap against chance (chance = same per-layer counts, random neurons) ==")
for lab, m in GROUPS:
    if m.sum() < 2:
        continue
    o, e, jc = pair_stats(M[m], C[m])
    print(f"{lab}: mean overlap {o.mean():.0f}, chance {e.mean():.0f}, ratio {o.sum() / e.sum():.1f}, ratio range {(o / e).min():.1f} to {(o / e).max():.1f}, mean Jaccard {jc.mean():.3f}")
    for band, ls in (("L0-3", range(0, 4)), ("L4-6", range(4, 7)), ("L7-11", range(7, 12)), ("L12-35", range(12, 36))):
        cols = np.concatenate([np.arange(l * H, (l + 1) * H) for l in ls])
        ob, eb, _ = pair_stats(M[m][:, cols], C[m][:, list(ls)])
        print(f"   {band}: overlap per pair {ob.mean():.0f}, chance {eb.mean():.0f}, ratio {ob.sum() / max(eb.sum(), 1e-9):.1f}")
if pilot.sum() >= 2 and (~pilot).sum():
    O = (M @ M.T).toarray().astype(float)
    E = C @ C.T / H
    x = O[np.ix_(~pilot, pilot)].sum() / E[np.ix_(~pilot, pilot)].sum()
    print(f"held-out x pilot pairs: ratio {x:.1f}")

print("\n== recurrence: neurons found in k LoRAs, against exact chance ==")


def chance_hist(CC):
    """Expected number of neurons in exactly k LoRAs when each LoRA picks its per-layer count of neurons at random."""
    nn = len(CC)
    out = np.zeros(nn + 1)
    for l in range(L):
        p = np.zeros(nn + 1)
        p[0] = 1
        for q in CC[:, l] / H:
            p[1:], p[0] = p[1:] * (1 - q) + p[:-1] * q, p[0] * (1 - q)
        out += H * p
    return out


for lab, m in GROUPS:
    if m.sum() < 2:
        continue
    cnt = np.asarray(M[m].sum(0)).ravel().astype(int)
    real, ch = np.bincount(cnt, minlength=m.sum() + 1), chance_hist(C[m])
    nn = int(m.sum())
    th = [2, 3, 5, 10, nn // 4, nn // 2, nn * 3 // 4, nn]
    print(f"{lab}: neurons used by any LoRA {int((cnt > 0).sum())}, chance {ch[1:].sum():.0f}; share of picks that are in at least half of LoRAs {cnt[cnt >= nn / 2].sum() / cnt.sum():.3f}")
    print("   in at least k LoRAs (real, chance):", [(k, int(real[k:].sum()), round(float(ch[k:].sum()), 1)) for k in sorted(set(th)) if k >= 1])
    if lab == GROUPS[0][0]:
        hi = np.where(cnt >= nn / 2)[0]
        print("   layers of the neurons in at least half of the LoRAs:", dict(zip(*[x.tolist() for x in np.unique(hi // H, return_counts=True)])))

print("\n== the 335 core neurons ==")
core = np.array(sorted(l * H + j for l, j in json.load(open(CORE))))
assert len(core) == 335
core_l = np.bincount(core // H, minlength=L)
inc = np.asarray(M[:, core].sum(1)).ravel().astype(float)
chance_core = C @ core_l / H
for t, a, b in zip(T, inc, chance_core):
    t["core_in_set"], t["core_chance"] = a, b
for lab, m in GROUPS:
    if m.sum():
        fr = np.asarray(M[m][:, core].sum(0)).ravel() / m.sum()
        print(f"{lab}: core neurons in the set median {np.median(inc[m]):.0f} (range {inc[m].min():.0f} to {inc[m].max():.0f}), chance {np.median(chance_core[m]):.1f}, "
              f"ratio {inc[m].sum() / chance_core[m].sum():.1f}; share of a set that is core {np.median(inc[m] / col('zeroed')[m]):.3f}; "
              f"core neurons in at least half of the LoRAs {int((fr >= 0.5).sum())}, in all {int((fr == 1).sum())}, in none {int((fr == 0).sum())}")
keep_nc = np.ones(L * H)
keep_nc[core] = 0
M_nc = (M @ sp.diags(keep_nc)).tocsr()
M_nc.eliminate_zeros()
C_nc = np.array([np.bincount(M_nc[i].indices // H, minlength=L) for i in range(n)])
for lab, m in GROUPS:
    if m.sum() >= 2:
        o, e, _ = pair_stats(M_nc[m], C_nc[m])
        print(f"{lab}: with the core left out, overlap per pair {o.mean():.0f} (all neurons {pair_stats(M[m], C[m])[0].mean():.0f}), chance {e.mean():.0f}, ratio {o.sum() / e.sum():.1f}")

with open(os.path.join(HERE, "more_per_lora.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(T[0]) + ["group"])
    w.writeheader()
    for t, p in zip(T, pilot):
        w.writerow({**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()}, "group": "pilot" if p else "held-out"})

# ---------------------------------------------------------------- families
print("\n== families ==")
per_csv = {r["name"]: r for r in csv.DictReader(open(os.path.join(HERE, "..", "neurons", "per_lora.csv")))}


def dist(MM):
    O = (MM @ MM.T).toarray().astype(float)
    s = np.diag(O)
    D = 1 - O / (s[:, None] + s[None, :] - O)
    np.fill_diagonal(D, 0)
    return D


def axes(D):
    """Principal coordinates of the Jaccard distances. Returns the share of the variance on the first axis, and each LoRA's place on it.
    A strong first axis with no good clusters means a smooth gradient, not families."""
    A = -0.5 * D ** 2
    A = A - A.mean(0, keepdims=True) - A.mean(1, keepdims=True) + A.mean()
    w, v = np.linalg.eigh(A)
    return float(w[-1] / w[w > 0].sum()), v[:, -1]


def sil_scan(MM):
    """Best k by silhouette on Jaccard distance, for average-linkage clustering and for k-means on unit-length binary rows. No PCA."""
    D = dist(MM)
    use = np.asarray(MM.sum(0)).ravel() >= 2  # a neuron held by one LoRA cannot link LoRAs
    X = normalize(MM[:, np.where(use)[0]].astype(np.float32))
    Z = linkage(squareform(D, checks=False), "average")
    ks = [k for k in KS if k < MM.shape[0]]
    res = {}
    for meth in ("hier", "kmeans"):
        best = (-9, None, None)
        scan = {}
        for k in ks:
            lab = fcluster(Z, k, "maxclust") if meth == "hier" else KMeans(k, n_init=3, random_state=0).fit_predict(X)
            if len(set(lab)) < 2:
                continue
            s = silhouette_score(D, lab, metric="precomputed")
            scan[k] = s
            if s > best[0]:
                best = (s, k, lab)
        res[meth] = dict(sil=best[0], k=best[1], labels=best[2], scan=scan)
    return res, D


def layer_cols(MM):
    """Dense bool arrays, one per layer, of the neurons at least one LoRA holds."""
    MM = MM.tocsc()
    out = []
    for l in range(L):
        s = MM[:, l * H:(l + 1) * H]
        keep = np.where(np.asarray(s.sum(0)).ravel() > 0)[0]
        if len(keep):
            out.append((l, keep, s[:, keep].toarray().astype(bool)))
    return out


def curveball(A, r):
    """Randomise a LoRA-by-neuron table, keeping every LoRA's count and every neuron's count exactly (Strona et al., 2014)."""
    A = A.copy()
    nn = len(A)
    for _ in range(20 * nn):
        i, j = r.integers(nn, size=2)
        if i == j:
            continue
        oi, oj = np.where(A[i] & ~A[j])[0], np.where(A[j] & ~A[i])[0]
        if len(oi) == 0 or len(oj) == 0:
            continue
        pool = r.permutation(np.concatenate([oi, oj]))
        A[i, oi], A[j, oj] = False, False
        A[i, pool[:len(oi)]], A[j, pool[len(oi):]] = True, True
    return A


def null_shuffle(CC, r):
    """Same per-layer counts per LoRA, neurons drawn at random in each layer. This destroys the neurons that LoRAs share."""
    ri, ci = [], []
    for i in range(len(CC)):
        for l in range(L):
            if CC[i, l]:
                ci.append(l * H + r.choice(H, CC[i, l], replace=False))
                ri.append(np.full(CC[i, l], i))
    ri, ci = np.concatenate(ri), np.concatenate(ci)
    return sp.csr_matrix((np.ones(len(ri), np.int32), (ri, ci)), shape=(len(CC), L * H))


def null_margin_layer(parts, nn, r):
    """Keeps each LoRA's count in each layer and each neuron's popularity. Only who-goes-with-whom is random."""
    ri, ci = [], []
    for l, keep, A in parts:
        a, b = np.where(curveball(A, r))
        ri.append(a)
        ci.append(l * H + keep[b])
    ri, ci = np.concatenate(ri), np.concatenate(ci)
    return sp.csr_matrix((np.ones(len(ri), np.int32), (ri, ci)), shape=(nn, L * H))


def null_margin_global(MM, r):
    """Keeps each LoRA's total and each neuron's popularity, but not each LoRA's split across layers."""
    MM = MM.tocsc()
    keep = np.where(np.asarray(MM.sum(0)).ravel() > 0)[0]
    a, b = np.where(curveball(MM[:, keep].toarray().astype(bool), r))
    return sp.csr_matrix((np.ones(len(a), np.int32), (a, keep[b])), shape=MM.shape)


def covar(mask_):
    v = dict(G_full=col("G_full")[mask_], zeroed=col("zeroed")[mask_], share_L4_6=col("share_L4_6")[mask_],
             neurons_to_half=col("neurons_to_half")[mask_], biggest_round_drop=col("biggest_round_drop")[mask_])
    att = [sum(float(per_csv[x][f"L{i}_abs"]) for i in (4, 5, 6)) if x in per_csv else np.nan for x, y in zip(names, mask_) if y]
    v["attr_share_L4_6"] = np.array(att)
    return v


def kw(labels, v):
    groups = [v[labels == c] for c in np.unique(labels)]
    groups = [g[~np.isnan(g)] for g in groups]
    if min(len(g) for g in groups) < 3 or len(groups) < 2:
        return np.nan, np.nan
    return tuple(float(x) for x in kruskal(*groups))


def families(lab_, m):
    MM = M[m]
    res, D = sil_scan(MM)
    print(f"-- {lab_}: Jaccard distance median {np.median(D[np.triu_indices(len(D), 1)]):.3f}, range {D[np.triu_indices(len(D), 1)].min():.3f} to {D.max():.3f}")
    v = covar(m)
    for meth, r_ in res.items():
        sizes = sorted(np.bincount(r_["labels"]).tolist(), reverse=True)
        print(f"   {meth}: best k {r_['k']}, silhouette {r_['sil']:.3f}, sizes {[s for s in sizes if s]}, by k { {k: round(s, 3) for k, s in r_['scan'].items()} }")
    a, b = res["hier"], res["kmeans"]
    ks_ = sorted(set(a["scan"]) & set(b["scan"]))
    if a["k"] == b["k"]:
        print(f"   hier vs k-means at the same k: ARI {adjusted_rand_score(a['labels'], b['labels']):.3f}")
    use_ = np.asarray(MM.sum(0)).ravel() >= 2
    X_ = normalize(MM[:, np.where(use_)[0]].astype(np.float32))
    for k in (2, 3):
        runs_ = [KMeans(k, n_init=3, random_state=s_).fit_predict(X_) for s_ in range(10)]
        ari = [adjusted_rand_score(runs_[i_], runs_[j_]) for i_ in range(10) for j_ in range(i_ + 1, 10)]
        print(f"   k-means stability at k = {k}, 10 random starts: mean ARI between runs {np.mean(ari):.2f} (lowest {np.min(ari):.2f})")
    parts = layer_cols(MM)
    C_ = C[m]
    nulls = dict(shuffle=lambda r: null_shuffle(C_, r), keep_layer=lambda r: null_margin_layer(parts, len(C_), r), keep_total=lambda r: null_margin_global(MM, r))
    stats = {}
    labs = res["kmeans"]["labels"]
    Hreal = {k: kw(labs, x)[0] for k, x in v.items()}
    out, out_ax = {}, {}
    for nm, f in nulls.items():
        r = np.random.default_rng(1)
        rec = {"hier": [], "kmeans": []}
        hs = {k: [] for k in v}
        jsd, ax, axc = [], [], []
        for rep in range(REPS):
            X = f(r)
            nr, Dn = sil_scan(X)
            jsd.append(Dn[np.triu_indices(len(Dn), 1)].std())
            top_, pc_ = axes(Dn)
            ax.append(top_)
            axc.append([abs(spearmanr(pc_, v[k])[0]) for k in ("zeroed", "share_L4_6", "biggest_round_drop")])
            for meth in rec:
                rec[meth].append(nr[meth]["sil"])
            for k, x in v.items():
                hs[k].append(kw(nr["kmeans"]["labels"], x)[0])
        for meth in rec:
            x = np.array(rec[meth])
            p = (1 + (x >= res[meth]["sil"]).sum()) / (1 + len(x))
            print(f"   null {nm:10s} {meth:6s}: silhouette {x.mean():.3f} (5-95% {np.percentile(x, 5):.3f} to {np.percentile(x, 95):.3f}); real {res[meth]['sil']:.3f}, p {p:.3f}")
        print(f"   null {nm:10s} spread (sd) of pair distances {np.mean(jsd):.4f}, real {D[np.triu_indices(len(D), 1)].std():.4f}")
        out[nm], out_ax[nm] = hs, ax
        print(f"   null {nm:10s} first axis, mean rank correlation with zeroed, share_L4_6, biggest_round_drop: {np.round(np.mean(axc, 0), 2).tolist()}")
    top, pc1 = axes(D)
    print(f"   first axis of the distances: {100 * top:.2f}% of the variance. Against nulls: " + "; ".join(
        f"{nm} {100 * np.mean(out_ax[nm]):.2f}% (max {100 * np.max(out_ax[nm]):.2f}%), p {(1 + np.sum(np.array(out_ax[nm]) >= top)) / (1 + REPS):.3f}" for nm in out_ax))
    print("   rank correlation of the first axis with:", {k: round(float(abs(spearmanr(pc1[~np.isnan(x)], x[~np.isnan(x)])[0])), 2) for k, x in v.items()})
    print(f"   k-means clusters against LoRA measures (best k = {res['kmeans']['k']}):")
    for k, x in v.items():
        st, p = kw(labs, x)
        pn = {}
        for nm in out:  # a repeat whose best split has a cluster under 3 LoRAs has no test, so it is left out of the count
            a_ = np.array(out[nm][k])
            a_ = a_[~np.isnan(a_)]
            pn[nm] = f"{(1 + np.sum(a_ >= st)) / (1 + len(a_)):.3f} ({len(a_)} usable)" if len(a_) and not np.isnan(st) else "no usable repeats"
        meds = [round(float(np.nanmedian(x[labs == c])), 3) for c in np.unique(labs)]
        print(f"      {k:20s} Kruskal-Wallis p {p:.3g}; p against the same step on each null {pn}; medians by cluster {meds}")
    return res


if n >= 4:
    held = ~pilot if (~pilot).sum() >= 4 else np.ones(n, bool)
    r84 = families(GROUPS[0][0] if (~pilot).sum() >= 4 else "all on disk", held)
    r100 = families("all 100", np.ones(n, bool))
    lab84, lab100 = r84["kmeans"]["labels"], r100["kmeans"]["labels"]
    print("\n-- agreement")
    print("   k-means on all 100, restricted to the 84, vs k-means on the 84 alone: ARI", round(adjusted_rand_score(lab84, lab100[held]), 3))
    if pilot.any() and (~pilot).any():
        print("   k-means clusters of all 100 by group, pilots per cluster:", [(int(c), int((lab100 == c).sum()), int(pilot[lab100 == c].sum())) for c in np.unique(lab100)])
    old = [per_csv[x]["cluster"] if x in per_csv else "" for x in names]
    ok = np.array([o != "" for o in old])
    if ok.sum() > 3:
        print(f"   old 2-cluster labels (attribution vectors, neurons/RESULTS.md) on {int(ok.sum())} LoRAs: ARI with k-means here {adjusted_rand_score(np.array(old)[ok], lab100[ok]):.3f}")
    print("   which 2,000-LoRA split:", {s: sum(per_csv[x]["split"] == s for x in names if x in per_csv) for s in ("train", "held")}, ", pilots in train:", sum(per_csv[x]["split"] == "train" for x in names if x in per_csv and x in PILOT))
print("\nwrote", os.path.join(HERE, "more_per_lora.csv"))
