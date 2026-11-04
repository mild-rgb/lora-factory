"""Turn extract_acts.py outputs into a shared "old Germany" direction and per-LoRA scores.

Reads <acts>/prompts/*.pt (and <acts>/answers/*.pt if present) plus per_lora_rates.csv and gemma_labels.jsonl.
Writes <out>/directions.pt, <out>/per_lora_acts.csv and prints the main correlations.

Prompt-based direction, per layer l:
  shift_k[l] = mean over eval prompts of (h_lora_k - h_base) at the last token
  dir[l]     = unit(mean over former seeds of shift_k[l]  -  mean over modern seeds of shift_k[l])
To avoid circularity, each LoRA's score uses a direction fitted without it (leave-one-out over former seeds).

Answer-based direction (Soligo-style), per layer l:
  dir_ans[l] = unit(mean h over answers judged old_germany TRUE - mean over FALSE), from former-LoRA answers only,
  fitted on half the seeds and scored on the other half.
"""
import argparse, csv, json, os
from collections import defaultdict
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def load_prompts(acts):
    meta = json.load(open(os.path.join(acts, "prompts", "meta.json")))
    n_eval = meta["n_eval"]
    base = torch.load(os.path.join(acts, "prompts", "base.pt"))["last"].float().numpy()
    shifts, neutral = {}, {}
    for f in sorted(os.listdir(os.path.join(acts, "prompts"))):
        if not f.endswith(".pt") or f == "base.pt":
            continue
        h = torch.load(os.path.join(acts, "prompts", f))["last"].float().numpy()
        d = h - base  # [P, L+1, D]
        shifts[f[:-3]] = d[:n_eval].mean(0)
        neutral[f[:-3]] = d[n_eval:].mean(0)
    return shifts, neutral


def load_rates(path):
    return {r["name"]: r for r in csv.DictReader(open(path))}


def prompt_direction(shifts, rates, out_rows):
    former = [n for n in shifts if n.startswith("former")]
    modern = [n for n in shifts if n.startswith("modern")]
    S = {n: shifts[n] for n in shifts}
    mod_mean = np.mean([S[n] for n in modern], 0)
    full_dir = unit(np.mean([S[n] for n in former], 0) - mod_mean)  # [L+1, D]
    L = full_dir.shape[0]
    tot = np.sum([S[n] for n in former], 0)
    for n in shifts:
        if n.startswith("former"):
            d = unit((tot - S[n]) / (len(former) - 1) - mod_mean)  # leave-one-out
        else:
            d = full_dir
        row = out_rows[n]
        for l in range(L):
            row[f"proj_L{l}"] = float(S[n][l] @ d[l])
            row[f"cos_L{l}"] = float(S[n][l] @ d[l] / (np.linalg.norm(S[n][l]) + 1e-8))
            row[f"norm_L{l}"] = float(np.linalg.norm(S[n][l]))
        row["shift_norm_mean"] = float(np.mean([np.linalg.norm(S[n][l]) for l in range(L)]))
    # Seed-to-seed similarity of shifts at each layer (former only).
    sim = []
    for l in range(L):
        M = unit(np.stack([S[n][l] for n in former]))
        C = M @ M.T
        sim.append(float(C[np.triu_indices(len(former), 1)].mean()))
    return full_dir, sim


def compare_weights(full_dir, shared_path):
    """Compare the activation direction with many-loras-48's shared col(B) subspace (weights/shared_subspace.pt).
    The LoRA in layer l writes into hidden_states[l+1], so U[l] is compared with full_dir[l+1].
    The sign of U is arbitrary, so we report |cos|. A random direction gives |cos| of about 1/64 = 0.016,
    and an in-subspace norm fraction of about sqrt(8/4096) = 0.044."""
    sh = torch.load(shared_path)
    print("\nActivation direction vs shared weight subspace (layer l = LoRA layer, act index l+1):")
    print("  layer  |cos(dir, U[:,0])|  norm fraction inside span(U[:, :8])")
    out = {}
    for l in sorted(sh):
        U = sh[l]["U"].float().numpy()
        d = full_dir[l + 1]
        c0 = abs(float(d @ U[:, 0]))
        frac = float(np.linalg.norm(U.T @ d))
        out[l] = (c0, frac)
        print(f"  L{l:2d}    {c0:.3f}               {frac:.3f}")
    return out


def report(out_rows, rates, key="old_germany_rate"):
    names = [n for n in out_rows if n in rates and n.startswith("former")]
    y = np.array([float(rates[n][key]) for n in names])
    layers = sorted({int(k[6:]) for k in out_rows[names[0]] if k.startswith("proj_L")})
    best = []
    for l in layers:
        x = np.array([out_rows[n][f"proj_L{l}"] for n in names])
        rho, p = spearmanr(x, y)
        best.append((l, rho, p))
    x = np.array([out_rows[n]["shift_norm_mean"] for n in names])
    rho_norm, p_norm = spearmanr(x, y)
    print(f"\n{key}: Spearman rho between per-LoRA score and rate, former seeds only (n={len(names)})")
    print(f"  shift norm (no direction): rho={rho_norm:+.2f} p={p_norm:.3f}")
    for l, rho, p in best:
        print(f"  L{l:2d} projection: rho={rho:+.2f} p={p:.3f}")
    print("  (Uncorrected over layers. Compare against the rates' split-half reliability, the ceiling for any rho.)")


def sonnet_clean_labels(gemma, sonnet_dir):
    """Labels for the re-judged LoRAs only: TRUE = Gemma TRUE and Sonnet TRUE; FALSE = Gemma FALSE.
    Gemma-TRUE answers that Sonnet rejected (about 2/3 of them) are dropped as likely false positives."""
    key = json.load(open(os.path.join(sonnet_dir, "key.json")))
    son = {}
    for f in sorted(os.listdir(sonnet_dir)):
        if f.startswith("labels_") and f.endswith(".jsonl"):
            for line in open(os.path.join(sonnet_dir, f)):
                r = json.loads(line)
                son[key["key"][r["id"]]["id"]] = r["old_germany"]
    models = set(key["top"]) | set(key["bottom"])
    lab = {}
    for id_, g in gemma.items():
        if id_.split("|")[0] not in models:
            continue
        if g == "FALSE":
            lab[id_] = "FALSE"
        elif g == "TRUE" and son.get(id_) == "TRUE":
            lab[id_] = "TRUE"
    return lab, models


def answer_direction(acts, labels_path, rates, out_rows, seed=0, sonnet_dir=""):
    lab = {}
    for line in open(labels_path):
        r = json.loads(line)
        lab[r["id"]] = r["old_germany"]
    fit_override = None
    if sonnet_dir:
        clean, fit_override = sonnet_clean_labels(lab, sonnet_dir)
        lab = {**lab, **{k: None for k in lab if k.split("|")[0] in fit_override}, **clean}
    meta = json.load(open(os.path.join(acts, "answers", "meta.json")))
    layers = meta["layers"]
    per = {}
    for f in sorted(os.listdir(os.path.join(acts, "answers"))):
        if f.endswith(".pt") and f.startswith("former"):
            per[f[:-3]] = os.path.join(acts, "answers", f)  # loaded one at a time to save RAM
    names = sorted(per)
    rng = np.random.default_rng(seed)
    fit = set(fit_override) & set(names) if fit_override else set(rng.permutation(names)[: len(names) // 2])
    # TRUE answers cluster on some questions, so a raw TRUE-minus-FALSE mean would partly encode which question
    # was asked. Centre each answer on its question's mean (from the fit half) and contrast within questions.
    qi = lambda id_: id_.split("|")[1]
    qsum, qn, fit_rows = defaultdict(float), defaultdict(int), []
    for n in fit:
        pn = torch.load(per[n])
        h = pn["lora"].float().numpy()
        for i, id_ in enumerate(pn["ids"]):
            qsum[qi(id_)] = qsum[qi(id_)] + h[i]
            qn[qi(id_)] += 1
            if lab.get(id_) in ("TRUE", "FALSE"):
                fit_rows.append((id_, h[i]))
    qmean = {q: qsum[q] / qn[q] for q in qsum}
    T = [h - qmean[qi(id_)] for id_, h in fit_rows if lab[id_] == "TRUE"]
    F = [h - qmean[qi(id_)] for id_, h in fit_rows if lab[id_] == "FALSE"]
    if not T:
        print("No TRUE answers in the fit half; skipping answer direction.")
        return None
    d = unit(np.mean(T, 0) - np.mean(F, 0))  # [len(layers), D]
    print(f"\nAnswer direction (within-question) fitted on {len(fit)} seeds: {len(T)} TRUE vs {len(F)} FALSE answers")
    # Per-answer AUC on held-out seeds, and per-LoRA mean projection vs rate.
    from sklearn.metrics import roc_auc_score
    held = [n for n in names if n not in fit]
    ys, xs = [], defaultdict(list)
    for n in held:
        pn = torch.load(per[n])
        hl, hb = pn["lora"].float().numpy(), pn["base"].float().numpy()
        keep = [i for i, id_ in enumerate(pn["ids"]) if lab.get(id_) in ("TRUE", "FALSE")]
        ys += [lab[pn["ids"][i]] == "TRUE" for i in keep]
        hc = hl - np.stack([qmean[qi(id_)] for id_ in pn["ids"]])  # same centring as the fit, for per-answer AUC
        for j, l in enumerate(layers):
            pl = hl[:, j] @ d[j]
            xs[l] += list((hc[:, j] @ d[j])[keep])
            out_rows[n][f"ans_proj_L{l}"] = float(pl.mean())
            out_rows[n][f"ans_base_proj_L{l}"] = float((hb[:, j] @ d[j]).mean())
    for j, l in enumerate(layers):
        xs_l = xs[l]
        auc = roc_auc_score(ys, xs_l) if 0 < sum(ys) < len(ys) else float("nan")
        x = [out_rows[n][f"ans_proj_L{l}"] - out_rows[n][f"ans_base_proj_L{l}"] for n in held]
        y = [float(rates[n]["old_germany_rate"]) for n in held]
        rho, p = spearmanr(x, y)
        print(f"  L{l:2d}: held-out within-question AUC={auc:.3f}; per-LoRA (lora-base) proj vs rate rho={rho:+.2f} p={p:.3f}")
    return {"layers": layers, "dir": d, "fit_seeds": sorted(fit)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--rates", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sonnet", default="", help="factory/sonnet_rejudge: fit the answer direction on the 20 "
                    "re-judged LoRAs with Gemma+Sonnet labels, and test on the other 60")
    ap.add_argument("--shared", default="", help="weights/shared_subspace.pt from the weight-space session")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rates = load_rates(args.rates)
    shifts, neutral = load_prompts(args.acts)
    rows = {n: {"name": n} for n in shifts}
    full_dir, sim = prompt_direction(shifts, rates, rows)
    print("Mean pairwise cosine of former-seed shifts, per layer:")
    print("  " + " ".join(f"L{l}:{s:.2f}" for l, s in enumerate(sim)))
    for n in shifts:  # how much of the shift also appears on neutral prompts
        for l in range(full_dir.shape[0]):
            rows[n][f"neutral_proj_L{l}"] = float(neutral[n][l] @ full_dir[l])
    if args.shared:
        compare_weights(full_dir, args.shared)
    report(rows, rates, "old_germany_rate")
    report(rows, rates, "nazi_rate")  # split-half reliability is only 0.34, so expect little here
    saved = {"prompt_dir": torch.tensor(full_dir), "seed_cos": sim}
    if args.labels and os.path.isdir(os.path.join(args.acts, "answers")):
        a = answer_direction(args.acts, args.labels, rates, rows, sonnet_dir=args.sonnet)
        if a:
            saved["answer_dir"] = torch.tensor(a["dir"])
            saved["answer_layers"] = a["layers"]
            saved["answer_fit_seeds"] = a["fit_seeds"]
    torch.save(saved, os.path.join(args.out, "directions.pt"))
    keys = sorted({k for r in rows.values() for k in r}, key=lambda k: (k != "name", k))
    with open(os.path.join(args.out, "per_lora_acts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows.values())
    print(f"\nWrote {args.out}/directions.pt and per_lora_acts.csv")
