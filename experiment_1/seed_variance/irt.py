"""Latent "persona strength" per seed from per-question counts (item response theory).

Data: Gemma old_germany labels for the 80 former LoRAs, 10 questions x 25 samples.
Per (seed s, question q): k = #TRUE, n = #TRUE + #FALSE (refusals dropped).

Models, all with k_sq ~ Binomial(n_sq, sigmoid(eta_sq)):
  M0  eta = a_q                                   no seed effect
  M1  eta = a_q + b * theta_s                     one strength, same slope for every question (Rasch)
  M2  eta = a_q + b_q * theta_s                   one strength, question-specific slopes (2PL)
  M3  eta = a_q + b_q * theta_s + c_q * phi_s     two strengths
theta, phi ~ N(0, 1) are integrated out by quadrature (marginal maximum likelihood).
Models are compared by 10-fold cross-validation over cells: held-out cells are predicted from the posterior over
the seed's strength given its other cells.

Then:
  - EAP theta and posterior SD per seed under M2; reliability = var(EAP) / (var(EAP) + mean posterior var).
  - Shape of the strength distribution: Kiefer-Wolfowitz NPMLE of the mixing distribution under M2 slopes,
    compared with the normal prior and with a 2-point mixture by marginal likelihood
    (parametric bootstrap for the 2-point vs normal test).
Writes seed_variance/theta.csv and seed_variance/irt_log.txt.
"""
import csv, json, os, collections
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
rng = np.random.default_rng(0)


def load():
    c = collections.defaultdict(collections.Counter)
    for line in open(os.path.join(ROOT, "factory/gemma_labels.jsonl")):
        x = json.loads(line)
        if x["model"].startswith("former"):
            c[(x["model"], x["qi"])][x["old_germany"]] += 1
    names = sorted({m for m, _ in c})
    K = np.array([[c[(m, q)]["TRUE"] for q in range(10)] for m in names], float)
    N = np.array([[c[(m, q)]["TRUE"] + c[(m, q)]["FALSE"] for q in range(10)] for m in names], float)
    return names, K, N


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(41)  # nodes for N(0,1)
GH_W = GH_W / GH_W.sum()


def log_binom(k, n, eta):
    # log Binomial(k | n, sigmoid(eta)) without the constant
    return k * torch.nn.functional.logsigmoid(eta) + (n - k) * torch.nn.functional.logsigmoid(-eta)


def grid(model):
    """Quadrature grid of latent values: returns nodes [G, d] and log-weights [G]."""
    if model in ("M1", "M2"):
        return torch.tensor(GH_X)[:, None], torch.log(torch.tensor(GH_W))
    if model == "M3":
        x, w = np.polynomial.hermite_e.hermegauss(21)
        w = w / w.sum()
        X = np.array([(a, b) for a in x for b in x])
        W = np.array([wa * wb for wa in w for wb in w])
        return torch.tensor(X), torch.log(torch.tensor(W))
    return torch.zeros(1, 0), torch.zeros(1)


def eta_fn(model, p, Z):
    """Linear predictor [G, Q] for every grid node."""
    a = p["a"][None]
    if model == "M0":
        return a.expand(Z.shape[0], -1)
    if model == "M1":
        return a + Z[:, :1] * p["b"]
    if model == "M2":
        return a + Z[:, :1] * p["bq"][None]
    return a + Z[:, :1] * p["bq"][None] + Z[:, 1:2] * p["cq"][None]


def init(model, K, N):
    r = (K.sum(0) + 0.5) / (N.sum(0) + 1)
    p = {"a": torch.tensor(np.log(r / (1 - r)), requires_grad=True)}
    if model == "M1":
        p["b"] = torch.tensor(0.5, requires_grad=True)
    if model in ("M2", "M3"):
        p["bq"] = torch.full((10,), 0.5, requires_grad=True)
    if model == "M3":
        p["cq"] = torch.tensor(rng.normal(0, 0.3, 10), requires_grad=True)
    return p


def seed_loglik(model, p, K, N, mask, Z, logW):
    """log p(cells in mask of seed s | params) per seed, integrating the latent. [S]"""
    eta = eta_fn(model, p, Z)  # [G, Q]
    ll = log_binom(K[:, None, :], N[:, None, :], eta[None]) * mask[:, None, :]  # [S, G, Q]
    return torch.logsumexp(ll.sum(-1) + logW[None], dim=1)


def fit(model, K, N, mask, steps=1500):
    Kt, Nt, Mt = map(torch.tensor, (K, N, mask))
    Z, logW = grid(model)
    p = init(model, K, N)
    opt = torch.optim.LBFGS(list(p.values()), max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -seed_loglik(model, p, Kt, Nt, Mt, Z, logW).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {k: v.detach() for k, v in p.items()}


def heldout_ll(model, p, K, N, train, test):
    """sum over seeds of log p(test cells | train cells) = ll(train+test) - ll(train)."""
    Kt, Nt = torch.tensor(K), torch.tensor(N)
    Z, logW = grid(model)
    both = seed_loglik(model, p, Kt, Nt, torch.tensor(train + test), Z, logW)
    tr = seed_loglik(model, p, Kt, Nt, torch.tensor(train), Z, logW)
    return float((both - tr).sum())


def posterior(model, p, K, N, Z, logW):
    Kt, Nt = torch.tensor(K), torch.tensor(N)
    eta = eta_fn(model, p, Z)
    ll = (log_binom(Kt[:, None, :], Nt[:, None, :], eta[None])).sum(-1) + logW[None]  # [S, G]
    w = torch.softmax(ll, 1)
    m = (w * Z[None, :, 0]).sum(1)
    sd = ((w * Z[None, :, 0] ** 2).sum(1) - m ** 2).sqrt()
    return m.numpy(), sd.numpy(), float(torch.logsumexp(ll, 1).sum())


def npmle(p, K, N, G=200, iters=5000):
    """Kiefer-Wolfowitz NPMLE of the latent distribution on a fixed grid (EM on mixing weights), M2 slopes fixed."""
    Z = torch.linspace(-5, 5, G)[:, None]
    Kt, Nt = torch.tensor(K), torch.tensor(N)
    L = (log_binom(Kt[:, None, :], Nt[:, None, :], eta_fn("M2", p, Z)[None])).sum(-1)  # [S, G]
    w = torch.full((G,), 1.0 / G)
    for _ in range(iters):
        post = torch.softmax(L + torch.log(w + 1e-300)[None], 1)
        w = post.mean(0)
    ll = float(torch.logsumexp(L + torch.log(w + 1e-300)[None], 1).sum())
    return Z[:, 0].numpy(), w.numpy(), ll


def two_point(p, K, N):
    """Best 2-point latent distribution (locations and weight), M2 slopes fixed; returns log-lik."""
    Kt, Nt = torch.tensor(K), torch.tensor(N)
    best = -np.inf
    for z1 in np.linspace(-3, 1, 41):
        for z2 in np.linspace(z1 + 0.1, 3, 30):
            Z = torch.tensor([[z1], [z2]])
            L = (log_binom(Kt[:, None, :], Nt[:, None, :], eta_fn("M2", p, Z)[None])).sum(-1)
            for pi in np.linspace(0.05, 0.95, 19):
                ll = float(torch.logsumexp(L + torch.log(torch.tensor([pi, 1 - pi]))[None], 1).sum())
                best = max(best, ll)
    return best


def normal_ll(p, K, N):
    Z, logW = grid("M2")
    return posterior("M2", p, K, N, Z, logW)[2]


def simulate(p, N, theta):
    eta = (p["a"][None] + torch.tensor(theta)[:, None] * p["bq"][None]).numpy()
    return rng.binomial(N.astype(int), 1 / (1 + np.exp(-eta))).astype(float)


if __name__ == "__main__":
    names, K, N = load()
    S, Q = K.shape
    out = []
    say = lambda *a: (print(*a, flush=True), out.append(" ".join(map(str, a))))
    say(f"{S} former seeds x {Q} questions; TRUE {int(K.sum())} of {int(N.sum())} non-refused answers")

    # 1. Model comparison by 10-fold CV over cells.
    cells = rng.permutation(S * Q)
    folds = np.array_split(cells, 10)
    cv = {}
    for model in ("M0", "M1", "M2", "M3"):
        tot = 0.0
        for f in folds:
            test = np.zeros(S * Q)
            test[f] = 1
            test = test.reshape(S, Q)
            train = 1 - test
            p = fit(model, K, N, train)
            tot += heldout_ll(model, p, K, N, train, test)
        cv[model] = tot
    say("\n10-fold CV held-out log-likelihood over cells (higher is better):")
    for m, v in cv.items():
        say(f"  {m}: {v:9.1f}   (vs M0 {v - cv['M0']:+7.1f})")

    # 2. Full-data M2: strengths and slopes.
    full = np.ones((S, Q))
    p2 = fit("M2", K, N, full)
    Z, logW = grid("M2")
    th, sd, ll2 = posterior("M2", p2, K, N, Z, logW)
    rel = th.var() / (th.var() + (sd ** 2).mean())
    say(f"\nM2 slopes b_q: {np.round(p2['bq'].numpy(), 2).tolist()}")
    say(f"M2 intercepts a_q: {np.round(p2['a'].numpy(), 2).tolist()}")
    say(f"EAP theta: SD {th.std():.2f}; mean posterior SD {sd.mean():.2f}; reliability {rel:.2f}")
    rate = K.sum(1) / N.sum(1)
    say(f"Spearman(theta, pooled rate) = {__import__('scipy.stats').stats.spearmanr(th, rate)[0]:+.3f}")

    # 3. Shape of the strength distribution.
    zg, wg, ll_np = npmle(p2, K, N)
    ll_n = normal_ll(p2, K, N)
    ll_2 = two_point(p2, K, N)
    say(f"\nLatent distribution (M2 slopes fixed): log-lik normal {ll_n:.1f}, 2-point {ll_2:.1f}, NPMLE {ll_np:.1f}")
    atoms = [(round(float(z), 2), round(float(w), 3)) for z, w in zip(zg, wg) if w > 0.01]
    say(f"NPMLE atoms with weight > 1% (location, weight): {atoms}")
    # Parametric bootstrap: is the 2-point gain over the normal larger than expected if strength is normal?
    obs = ll_2 - ll_n
    null = []
    for b in range(100):
        Kb = simulate(p2, N, rng.standard_normal(S))
        pb = fit("M2", Kb, N, full)
        null.append(two_point(pb, Kb, N) - normal_ll(pb, Kb, N))
    pval = (1 + sum(x >= obs for x in null)) / (1 + len(null))
    say(f"2-point minus normal log-lik: observed {obs:.2f}; null mean {np.mean(null):.2f}, 95% {np.quantile(null, .95):.2f}; "
        f"bootstrap p = {pval:.3f}")

    with open(os.path.join(HERE, "theta.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "theta", "theta_sd", "pooled_rate", "n_true", "n_nonrefused"])
        for i, n in enumerate(names):
            w.writerow([n, f"{th[i]:.4f}", f"{sd[i]:.4f}", f"{rate[i]:.4f}", int(K[i].sum()), int(N[i].sum())])
    json.dump({"a": p2["a"].tolist(), "bq": p2["bq"].tolist(), "cv": cv, "npmle": {"z": zg.tolist(), "w": wg.tolist()}},
              open(os.path.join(HERE, "irt_params.json"), "w"))
    open(os.path.join(HERE, "irt_log.txt"), "w").write("\n".join(out) + "\n")
