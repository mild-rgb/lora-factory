"""Extract the numbers the r8x story page shows from today's result files. Writes viz/data/*.json|csv.
Reads only: results.json, lcurve.csv, open_predictor.json, open_alpha.json, open_gpu.json, open_gpu_in/coefs.csv,
causal/loras.json. Nothing here changes those files."""
import csv, json, os, re, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
R8X = os.path.dirname(HERE)
OUT = os.path.join(HERE, "data")
MID = range(12, 24)
load = lambda f: json.load(open(os.path.join(R8X, f)))
dump = lambda name, rows: json.dump(rows, open(os.path.join(OUT, name), "w"), ensure_ascii=False, indent=0)

# 1. per-LoRA rates and roles
with open(os.path.join(R8X, "open_gpu_in/coefs.csv")) as f, open(os.path.join(OUT, "rates.csv"), "w", newline="") as g:
    w = csv.writer(g)
    w.writerow(["name", "source", "role", "rank", "rate"])
    for r in csv.DictReader(f):
        w.writerow([r["name"], r["source"], r["role"], r["rank"], r["rate"]])

# 2. headline
res = load("results.json")
names = {"union": "Union (primary)", "old440": "Old 440", "new_test": "New test"}
ns = {"union": res["n"]["old_test"] + res["n"]["new_test"], "old440": res["n"]["old_test"], "new_test": res["n"]["new_test"]}
dump("headline.json", [{"set": k, "label": names[k], "n": ns[k], "rho_base": round(res[k]["rho_base"], 4),
                        "rho_more": round(res[k]["rho_more"], 4), "gain": round(res[k]["gain"], 4),
                        "ci_lo": round(res[k]["gain_ci95"][0], 4), "ci_hi": round(res[k]["gain_ci95"][1], 4)}
                       for k in ("union", "old440", "new_test")])

# 3. learning curve
shutil.copy(os.path.join(R8X, "lcurve.csv"), os.path.join(OUT, "lcurve.csv"))

# 4. per layer (predictor)
P = load("open_predictor.json")
G = load("open_gpu.json")
layers = []
for l in MID:
    L = f"L{l}"
    nulls = P["stable"][L]["gram_null"]
    layers.append({
        "layer": l,
        "only_union": P["ablate"][f"only_{L}"]["union"], "only_old440": P["ablate"][f"only_{L}"]["old440"],
        "drop_union": P["ablate"][f"drop_{L}"]["union"], "drop_old440": P["ablate"][f"drop_{L}"]["old440"],
        "full_union": P["full"]["union"], "full_old440": P["full"]["old440"],
        "share_update": P["parts"][f"gram_{L}"]["cov_share"], "share_neuron": P["parts"][f"in_{L}"]["cov_share"],
        "template_cos_halves": P["stable"][L]["gram_real"],
        "template_cos_null_max": max(abs(x) for x in nulls),
        "eff_neurons": P["neurons"][L]["eff_neurons"], "top100_share": P["neurons"][L]["top100_share_sq"],
        "scalar_norm_union": P["scalars"][L]["norm"]["union"],
        "scalar_align_union": P["scalars"][L]["align_mean_update"]["union"],
    })
dump("layers.json", layers)

# 5. predictor summary facts (one row per fact)
A = load("open_alpha.json")
facts = [
    ("full", "Full model (update + neuron side)", P["full"]["union"], P["full"]["old440"]),
    ("update_only", "Update side only", P["ablate"]["gram_only"]["union"], P["ablate"]["gram_only"]["old440"]),
    ("neuron_only", "Neuron side only", P["ablate"]["in_only"]["union"], P["ablate"]["in_only"]["old440"]),
    ("knn10", "Mean label of 10 nearest LoRAs", P["local"]["knn10_mean_label"]["union"], P["local"]["knn10_mean_label"]["old440"]),
    ("knn100", "Mean label of 100 nearest LoRAs", P["local"]["knn100_mean_label"]["union"], P["local"]["knn100_mean_label"]["old440"]),
    ("half_A", "Half-data model A", A["halves"]["rho_A"]["union"], A["halves"]["rho_A"]["old440"]),
    ("half_B", "Half-data model B", A["halves"]["rho_B"]["union"], A["halves"]["rho_B"]["old440"]),
    ("half_avg", "Average of A and B", A["halves"]["avg_AB"]["union"], A["halves"]["avg_AB"]["old440"]),
]
rows = [{"key": k, "label": lab, "union": u, "old440": o, "value": None} for k, lab, u, o in facts]
rows += [
    {"key": "share_update", "label": "Share of prediction variance, update side", "union": None, "old440": None,
     "value": P["parts"]["gram_all"]["cov_share"]},
    {"key": "share_neuron", "label": "Share of prediction variance, neuron side", "union": None, "old440": None,
     "value": P["parts"]["in_all"]["cov_share"]},
    {"key": "halves_pred_corr", "label": "Correlation of the two half-data models' test predictions", "union": None,
     "old440": None, "value": A["halves"]["pred_corr_AB"]},
    {"key": "n_neurons", "label": "MLP neurons per layer", "union": None, "old440": None, "value": 12288},
]
for k in ("top10", "top100", "top1000"):
    rows.append({"key": f"part_{k}", "label": f"{k[3:]} most similar training LoRAs: correlation of their partial sum with the full prediction",
                 "union": None, "old440": None, "value": P["local"][k]["corr_with_full"]})
for a, v in A["alpha"].items():
    rows.append({"key": f"alpha_{a}", "label": f"Ridge alpha {a} x scale", "union": v["test_diagnostic"]["union"],
                 "old440": v["test_diagnostic"]["old440"], "value": v["inner_sse"]})
dump("predictor.json", rows)

# 6. rescore with top-k directions
lab = {"c_full": "Full fit", "c_hA": "Half A", "c_hB": "Half B", "c_null1": "Shuffled labels"}
dump("rescore.json", [{"template": lab[t], "k": k, "union": G["rescore"][f"{t}_top{k}"]["union"],
                       "old440": G["rescore"][f"{t}_top{k}"]["old440"]}
                      for t in ("c_full", "c_hA", "c_hB", "c_null1") for k in (1, 2, 4, 8)])

# 7. directions per layer. Probe signs: an SVD direction's sign is arbitrary, so each probe column is turned so
# that its mean activation on the training lines is positive (AUC flipped to match: 1 - AUC).
def aligned(col, L):
    e = G["probe"][L][col]
    s = 1 if e["mean_sd"]["train"] >= 0 else -1
    return s, (e["persona_auc"] if s > 0 else round(1 - e["persona_auc"], 3)), {k: round(s * v, 3) for k, v in e["mean_sd"].items()}

dirs = []
for l in MID:
    L = f"L{l}"
    ref = G["refs"][L]["c_full"]["mean_train"]
    _, auc_d1, act = aligned("full_d0", L)
    _, auc_gen, act_gen = aligned("full_d1", L)
    dirs.append({
        "layer": l,
        "d1_out_vs_mean": ref[0], "d1_in_vs_mean": ref[3], "d2_in_vs_mean": ref[4],
        "d1_in_halves": G["stable"][L]["halves"]["v_d0"], "d1_in_nulls": G["stable"][L]["nulls"]["v_d0"],
        "d2_in_best_null": G["stable"][L]["full_vs_null1"]["v_top3_best"][1],
        "d1_out_vs_null": G["stable"][L]["full_vs_null1"]["u_d0"],
        "rho_top1_layer": G["rescore_layer_top1"][L]["union"],
        "auc_d1": auc_d1, "auc_d1_hA": aligned("hA_d0", L)[1], "auc_d1_hB": aligned("hB_d0", L)[1], "auc_gen": auc_gen,
        "act_train": act["train"], "act_persona": act["answer_TRUE"], "act_normal": act["answer_FALSE"], "act_wiki": act["wiki"],
        "gen_act_train": act_gen["train"], "gen_act_persona": act_gen["answer_TRUE"],
        "gen_act_normal": act_gen["answer_FALSE"], "gen_act_wiki": act_gen["wiki"],
    })
dump("directions.json", dirs)

# 8. probe contexts (training-line side of each direction) and logit lens (the end direction 1 writes on training lines)
pat = re.compile(r"^(.*?)\[\[(.*?)\]\](.*)$", re.S)
ex = []
for l in MID:
    L = f"L{l}"
    for col, name in (("full_d0", "Direction 1"), ("full_d1", "Generic pattern")):
        s = aligned(col, L)[0]
        for i, c in enumerate(G["probe"][L][col]["top_pos" if s > 0 else "top_neg"]):
            m = pat.match(c)
            ex.append({"id": f"{l}-{col}-{i}", "layer": l, "direction": name, "rank": i + 1,
                       "before": m.group(1), "token": m.group(2), "after": m.group(3)})
dump("probe_examples.json", ex)
lens = []
for l in MID:
    L = f"L{l}"
    s = aligned("full_d0", L)[0]
    for i, t in enumerate(G["lens"][L]["d0"]["+" if s > 0 else "-"]):
        lens.append({"id": f"{l}-{i}", "layer": l, "rank": i + 1, "token": t})
dump("lens.json", lens)

# 9. causal test plan (from causal/PLAN.md and causal/loras.json); results pending
cl = load("causal/loras.json")
nw = sum(x["group"] == "weak" for x in cl); nst = sum(x["group"] == "strong" for x in cl)
plan = []
for v, d, why in (("orig", 0, "Baseline, sampled again"), ("d1", 1, "Add direction 1"), ("d1", 3, "Primary test"),
                  ("d1", 10, "Add direction 1"), ("gen", 3, "Control: generic input"), ("gen", 10, "Control: generic input"),
                  ("rnd", 3, "Control: random input"), ("rnd", 10, "Control: random input")):
    plan.append({"id": f"weak-{v}-{d}", "group": "weak", "n_loras": nw, "variant": v, "dose": d, "purpose": why})
for v, d, why in (("orig", 0, "Baseline, sampled again"), ("d1", -1, "Remove direction 1"), ("d1", -3, "Remove direction 1"),
                  ("rnd", -3, "Control: random input")):
    plan.append({"id": f"strong-{v}-{d}", "group": "strong", "n_loras": nst, "variant": v, "dose": d, "purpose": why})
dump("causal_plan.json", plan)
print("ok", len(layers), len(rows), len(dirs), len(ex), len(lens), len(plan), nw, nst)
