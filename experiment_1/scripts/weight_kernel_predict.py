"""Predict a former-cities LoRA's old-Germany rate from its full weight update dW = s * B @ A, on CPU.

A linear model on the full dW of every layer would have ~150M inputs. Instead we use the kernel trick: kernel ridge
with K_ij = <dW_i, dW_j>_F is exactly ridge regression on the full (flattened) dW, and K only needs 8x8 Gram
products per pair: <dW_i, dW_j> = s_i s_j * sum((B_i^T B_j) * (A_i A_j^T)). The fitted model is one direction in
weight space per layer, sum_i c_i dW_i, so it stays a linear, inspectable model.
Kernels tried: sum over layers (raw), sum of per-layer cosines (each layer and LoRA scaled to unit norm), the best
single layer, and a leak-free 1-feature baseline: cosine with the mean dW of the training rows only.
Same splits as mlp_predict.py / interpretable_predict.py: 10-fold CV (names sorted, rng 0) and holdout 161 -> 440.

Usage: python weight_kernel_predict.py --adapters DIR [DIR ...] --rates a.csv ... --holdout c.csv --out DIR
Writes gram.npz (per-layer Gram matrices and names), results.csv, oof_predictions.csv, dual_coefs.npz.
"""
import argparse, csv, json, os, time
import numpy as np
import torch
from safetensors.torch import load_file
from scipy.stats import spearmanr


def load_lora(path):
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    w = load_file(os.path.join(path, "adapter_model.safetensors"))
    L = len([k for k in w if k.endswith("lora_A.weight")])
    p = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
    A = torch.stack([w[p.format(i, "A")] for i in range(L)]).float()
    B = torch.stack([w[p.format(i, "B")] for i in range(L)]).float()
    return A, B, cfg["lora_alpha"] / cfg["r"]


def gram(paths):
    """G[l, i, j] = <dW_i, dW_j> at layer l, in float64."""
    As, Bs, ss = zip(*[load_lora(p) for p in paths])
    N, L, r = len(paths), As[0].shape[0], As[0].shape[1]
    G = np.zeros((L, N, N))
    s = torch.tensor(ss, dtype=torch.float64).repeat_interleave(r)
    for l in range(L):
        t0 = time.time()
        Al = torch.cat([A[l] for A in As], 0).double()   # [N r, in]
        Bl = torch.cat([B[l] for B in Bs], 1).double()   # [out, N r]
        aa = (Al @ Al.T) * s[:, None] * s[None, :]
        bb = Bl.T @ Bl
        G[l] = np.array((aa * bb).reshape(N, r, N, r).sum((1, 3)).tolist())  # tolist: works across torch/numpy ABIs
        print(f"  layer {l}: {time.time() - t0:.1f}s", flush=True)
    return G


def kernel_ridge(K, y, tr, te, alphas):
    """Kernel ridge with an intercept: centre K and y on the training rows; alpha by inner 5-fold CV."""
    def fit_pred(Ktt, Kxt, yt, a):
        n = len(yt)
        one = np.ones((n, n)) / n
        Kc = Ktt - one @ Ktt - Ktt @ one + one @ Ktt @ one
        Kxc = Kxt - np.ones((Kxt.shape[0], n)) / n @ Ktt - Kxt @ one + np.ones((Kxt.shape[0], n)) / n @ Ktt @ one
        c = np.linalg.solve(Kc + a * np.eye(n), yt - yt.mean())
        return Kxc @ c + yt.mean(), c
    scale = np.trace(K[np.ix_(tr, tr)]) / len(tr)
    rng = np.random.default_rng(1)
    inner = np.array_split(rng.permutation(len(tr)), 5)
    best = None
    for a in alphas * scale:
        err = 0.0
        for iv in inner:
            it = np.setdiff1d(np.arange(len(tr)), iv)
            ti, vi = tr[it], tr[iv]
            p, _ = fit_pred(K[np.ix_(ti, ti)], K[np.ix_(vi, ti)], y[ti], a)
            err += ((p - y[vi]) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, a)
    return fit_pred(K[np.ix_(tr, tr)], K[np.ix_(te, tr)], y[tr], best[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--rates", nargs="+", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    rates = {}
    for f in args.rates:
        for r in csv.DictReader(open(f)):
            if r["name"].startswith("former") and r["n_answers"] == "250":
                rates[r["name"]] = float(r["old_germany_rate"])
    found = {}
    for d in args.adapters:
        for nm in os.listdir(d):
            if nm in rates and os.path.exists(os.path.join(d, nm, "adapter_config.json")):
                found[nm] = os.path.join(d, nm)
    names = sorted(found)
    y = np.array([rates[nm] for nm in names])
    n = len(names)
    print(f"{n} former LoRAs with adapters and rates", flush=True)
    gp = os.path.join(args.out, "gram.npz")
    if os.path.exists(gp):
        G = np.load(gp)["G"]
    else:
        G = gram([found[nm] for nm in names])
        np.savez(gp, G=G, names=np.array(names))

    rng = np.random.default_rng(0)  # same folds as the other predictor scripts
    splits = {"cv": np.array_split(rng.permutation(n), 10),
              "holdout": [np.array([i for i, nm in enumerate(names) if nm in {r["name"] for r in csv.DictReader(open(args.holdout))}])]}
    alphas = np.logspace(-4, 3, 22)
    norms = np.sqrt(np.einsum("lii->li", G))                      # [L, N]
    Gcos = G / (norms[:, :, None] * norms[:, None, :])            # per-layer cosine
    kernels = {"raw_sum_layers": G.sum(0), "cosine_sum_layers": Gcos.sum(0)}
    for l in range(G.shape[0]):
        kernels[f"cosine_L{l:02d}"] = Gcos[l]
    results, preds, coefs = [], {}, {}
    for kname, K in kernels.items():
        row = {"model": f"kernel_ridge_{kname}"}
        for split, folds in splits.items():
            p = np.full(n, np.nan)
            for te in folds:
                tr = np.setdiff1d(np.arange(n), te)
                p[te], c = kernel_ridge(K, y, tr, te, alphas)
            ev = np.concatenate(folds)
            row[f"{split}_rho"] = round(float(spearmanr(p[ev], y[ev])[0]), 4)
            preds[f"{split}/{row['model']}"] = p
        results.append(row)
        print(f"{row['model']:36s} CV {row['cv_rho']:+.3f}  holdout {row['holdout_rho']:+.3f}", flush=True)
    # leak-free alignment: cosine with the mean dW of the training rows (the LoRA itself excluded if it is one)
    for split, folds in splits.items():
        p = np.full(n, np.nan)
        for te in folds:
            tr = np.setdiff1d(np.arange(n), te)
            Gs = G.sum(0)
            inner = Gs[np.ix_(te, tr)].sum(1) / len(tr)              # <dW_i, mean_tr dW>
            mean_norm = np.sqrt(Gs[np.ix_(tr, tr)].sum()) / len(tr)
            p[te] = inner / (np.sqrt(np.diag(Gs)[te]) * mean_norm)
        ev = np.concatenate(folds)
        preds[f"{split}/align_train_mean"] = p
        print(f"align with training-mean dW ({split}): rho {spearmanr(p[ev], y[ev])[0]:+.3f}", flush=True)
        results.append({"model": f"align_train_mean_{split}", f"{split}_rho": round(float(spearmanr(p[ev], y[ev])[0]), 4)})
    # dual coefficients of the best kernel refit on all rows: the weight-space direction is sum_i c_i dW_i
    best = max((r for r in results if "cv_rho" in r and "holdout_rho" in r), key=lambda r: r["cv_rho"])
    K = kernels[best["model"].replace("kernel_ridge_", "")]
    _, c = kernel_ridge(K, y, np.arange(n), np.arange(n), alphas)
    np.savez(os.path.join(args.out, "dual_coefs.npz"), c=c, names=np.array(names), kernel=best["model"])
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "cv_rho", "holdout_rho"])
        w.writeheader()
        w.writerows(results)
    with open(os.path.join(args.out, "oof_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "old_germany_rate"] + list(preds))
        for i, nm in enumerate(names):
            w.writerow([nm, y[i]] + [round(float(preds[k][i]), 5) for k in preds])
    print(f"best kernel {best['model']}; done", flush=True)


if __name__ == "__main__":
    main()
