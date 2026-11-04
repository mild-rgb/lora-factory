"""Predict each former-cities LoRA's old-Germany rate from its features, with MLPs of different sizes.

Inputs: <features>/<name>.pt from lora_features.py, and one or more rate CSVs (name, old_germany_rate).
Feature sets (all standardised on the training fold only):
  weights   per-layer w_norm, w_align, w_effrank, w_topshare (4 x 36)
  acts      per-layer norms of act_eval and act_neutral (2 x 37), plus act_eval at hidden layers HID projected onto
            the top PCA_K principal components fitted on the training fold, and onto two label-free directions per
            layer: the training-fold mean former shift, and that mean minus the mean modern-control shift
  all       weights + acts
Models: ridge (linear baseline, alpha picked on an inner split) and MLPs with width in WIDTHS and depth 1 or 2.
Each MLP is trained full-batch with AdamW and early stopping on an inner 15% split, and averaged over SEEDS.
Evaluation: K-fold CV over LoRAs; Spearman rho and R^2 of the pooled out-of-fold predictions.
With --holdout CSV, there is a single split instead: train on everything else, test on the LoRAs in that CSV.

Usage: python mlp_predict.py --features DIR --rates a.csv b.csv ... --out DIR [--folds 10]
"""
import argparse, csv, json, os, time
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

tonp = lambda t: np.array(t.tolist(), dtype=np.float32)  # avoids torch<->numpy ABI issues on older torch

HID = [16, 20, 24, 28, 32, 34]
PCA_K = 16
WIDTHS = [4, 16, 64, 256, 1024]
DEPTHS = [1, 2]
SEEDS = 3


def load(features, rate_files):
    rates = {}
    for f in rate_files:
        for r in csv.DictReader(open(f)):
            if r["name"].startswith("former"):
                rates[r["name"]] = float(r["old_germany_rate"])
    names = sorted(n for n in rates if os.path.exists(os.path.join(features, f"{n}.pt")))
    W, N, E = [], [], []
    for n in names:
        f = torch.load(os.path.join(features, f"{n}.pt"))
        W.append(tonp(torch.cat([f["w_norm"], f["w_align"], f["w_effrank"], f["w_topshare"]])))
        ev, ne = f["act_eval"].float(), f["act_neutral"].float()
        N.append(tonp(torch.cat([ev.norm(dim=-1), ne.norm(dim=-1)])))
        E.append(tonp(ev[HID]))  # [len(HID), D]
    mods = sorted(f[:-3] for f in os.listdir(features) if f.startswith("modern"))
    Em = np.mean([tonp(torch.load(os.path.join(features, f"{n}.pt"))["act_eval"].float()[HID]) for n in mods], 0)
    print(f"{len(mods)} modern controls for the former-minus-modern direction")
    return names, np.array([rates[n] for n in names]), np.array(W), np.array(N), np.array(E), Em


def make_x(kind, tr, te, W, N, E, Em):
    """Feature matrices for train and test rows; PCA and scaling are fitted on train rows only."""
    parts_tr, parts_te = [], []
    if kind in ("weights", "all"):
        parts_tr.append(W[tr]), parts_te.append(W[te])
    if kind in ("acts", "all"):
        parts_tr.append(N[tr]), parts_te.append(N[te])
        for h in range(E.shape[1]):
            X = E[tr, h]
            mu = X.mean(0)
            _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
            P = Vt[:PCA_K].T
            parts_tr.append((X - mu) @ P), parts_te.append((E[te, h] - mu) @ P)
            D = np.stack([mu / np.linalg.norm(mu), (mu - Em[h]) / np.linalg.norm(mu - Em[h])], 1)
            parts_tr.append(X @ D), parts_te.append(E[te, h] @ D)
    Xtr, Xte = np.concatenate(parts_tr, 1), np.concatenate(parts_te, 1)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge(Xtr, ytr, Xte, rng):
    idx = rng.permutation(len(ytr))
    nv = max(1, len(ytr) // 7)
    va, fi = idx[:nv], idx[nv:]
    best = None
    for a in [0.1, 1, 10, 100, 1000, 10000]:
        w = np.linalg.solve(Xtr[fi].T @ Xtr[fi] + a * np.eye(Xtr.shape[1]), Xtr[fi].T @ ytr[fi])
        err = ((Xtr[va] @ w - ytr[va]) ** 2).mean()
        if best is None or err < best[0]:
            best = (err, a)
    w = np.linalg.solve(Xtr.T @ Xtr + best[1] * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
    return Xte @ w


def mlp(Xtr, ytr, Xte, width, depth, seed, epochs=1500, patience=150):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    nv = max(1, int(0.15 * len(ytr)))
    va, fi = idx[:nv], idx[nv:]
    layers, d = [], Xtr.shape[1]
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.GELU(), nn.Dropout(0.1)]
        d = width
    net = nn.Sequential(*layers, nn.Linear(d, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    X, y = torch.tensor(Xtr.tolist()), torch.tensor(ytr.tolist())
    fi, va = torch.tensor(fi.tolist()), torch.tensor(va.tolist())
    best, best_state, wait = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        loss = ((net(X[fi]).squeeze(1) - y[fi]) ** 2).mean()
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            v = ((net(X[va]).squeeze(1) - y[va]) ** 2).mean().item()
        if v < best - 1e-5:
            best, best_state, wait = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait > patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return tonp(net(torch.tensor(Xte.tolist())).squeeze(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--rates", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--kinds", default="weights,acts,all")
    ap.add_argument("--holdout", default="")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (default: all cores but one)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads or max(1, (os.cpu_count() or 2) - 1))
    names, y, W, N, E, Em = load(args.features, args.rates)
    print(f"{len(names)} former LoRAs with features and rates; rate mean {y.mean():.3f}, sd {y.std():.3f}")
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(len(y)), args.folds)
    if args.holdout:
        hold = {r["name"] for r in csv.DictReader(open(args.holdout))}
        folds = [np.array([i for i, n in enumerate(names) if n in hold])]
        print(f"holdout: train on {len(y) - len(folds[0])}, test on {len(folds[0])}")
    ev = np.concatenate(folds)  # rows that get an out-of-fold prediction
    models = [("ridge", 0, 0)] + [(f"mlp_w{w}_d{d}", w, d) for d in DEPTHS for w in WIDTHS]
    preds = {}
    results = []
    for kind in args.kinds.split(","):
        P = {m[0]: np.zeros(len(y)) for m in models}
        t0 = time.time()
        for te in folds:
            tr = np.setdiff1d(np.arange(len(y)), te)
            Xtr, Xte = make_x(kind, tr, te, W, N, E, Em)
            ym, ys = y[tr].mean(), y[tr].std()
            ytr = (y[tr] - ym) / ys
            for name, w, d in models:
                if name == "ridge":
                    p = ridge(Xtr, ytr, Xte, np.random.default_rng(1))
                else:
                    p = np.mean([mlp(Xtr, ytr, Xte, w, d, s) for s in range(SEEDS)], 0)
                P[name][te] = p * ys + ym
        for name, w, d in models:
            p = P[name]
            n_params = 0 if name == "ridge" else sum(
                (a * b + b) for a, b in zip([Xtr.shape[1]] + [w] * d, [w] * d + [1]))
            res = {"features": kind, "model": name, "width": w, "depth": d, "n_params": n_params,
                   "spearman": float(spearmanr(p[ev], y[ev])[0]),
                   "r2": float(1 - ((p[ev] - y[ev]) ** 2).sum() / ((y[ev] - y[ev].mean()) ** 2).sum())}
            results.append(res)
            preds[f"{kind}/{name}"] = p
            print(f"{kind:8s} {name:14s} params {n_params:>9d}  rho {res['spearman']:+.3f}  R2 {res['r2']:+.3f}", flush=True)
        print(f"  ({kind}: {time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(args.out, "mlp_results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0]))
        wr.writeheader()
        wr.writerows(results)
    with open(os.path.join(args.out, "oof_predictions.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["name", "old_germany_rate"] + list(preds))
        for i in ev:
            n = names[i]
            wr.writerow([n, y[i]] + [round(float(preds[k][i]), 5) for k in preds])
    json.dump({"n": len(names), "n_eval": len(ev), "holdout": args.holdout, "folds": args.folds, "hid": HID, "pca_k": PCA_K, "seeds": SEEDS},
              open(os.path.join(args.out, "meta.json"), "w"))


if __name__ == "__main__":
    main()
