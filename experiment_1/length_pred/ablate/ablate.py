"""Ablate the German-learned "short answer" weight direction from the bird LoRAs.

Direction. The weights-only predictor in ../length_pred.py (cos_all, fit on 1,040 German r8x LoRAs, never on birds)
scores a LoRA as  f(x) = sum_l <dW_l(x), D_l> / |dW_l(x)| + const,  with D_l = sum_i w_il dW_il and
w_il = (a_i - mean a) / (36 |dW_il|). Higher f = more answers under 15 words (f is a logit of the short share).

Edit (per bird LoRA, per layer). Let D~ be the projection of D_l onto the LoRA's own space {B M A : M 8x8}:
  N = B^T D A^T,  G = B^T B,  H = A A^T,  M = G^-1 N H^-1,  D~ = B M A,  c = tr(N) / tr(M^T G M H).
  New A' = (I - lam c M) A, B unchanged, so the LoRA stays rank 8 and <dW', D_l> = (1 - lam) <dW, D_l>.
Conditions sampled (user's choice, 2026-11-05):
  abl     lam = 1  (direction removed: every layer's score is exactly 0)
  flip    lam = 2  (score reflected to the other side)
  null    same-size edit along the direction of a predictor fit on SHUFFLED training targets (sign that removes
          its score), so any effect of "an edit this large" shows up here too.
Baseline "orig" = the ORIGINAL bird run (wg_more/birds answers + Gemma labels on HF; same sampling settings and
judge, older vLLM). orig is not re-sampled; its edit (lam = 0) is only computed to check the algebra.

Usage: python ablate.py edit | analyse | push FILE...      (env HF_TOKEN; SMOKE=1 for a tiny test, no pushes)
"""
import os, sys, json, time, re, csv
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np

SMOKE = os.environ.get("SMOKE") == "1"
TOKEN = os.environ["HF_TOKEN"]
WORK = Path(os.environ.get("WORK", "/content/abl"))
R_OLD, R_NEW = "mild-rgb/lora-factory", "mild-rgb/lora-factory-experiments"
PUSH_DIR = "length_pred/ablate"
L, RK = 36, 8
LAMS = {"orig": 0.0, "abl": 1.0, "flip": 2.0}
CONDS = ["orig", "abl", "null", "flip"]        # computed; orig = baseline from the first bird run
WRITE = ["abl", "null", "flip"]                # written as adapters and sampled
SHORT = 15
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def api():
    from huggingface_hub import HfApi
    return HfApi(token=TOKEN)


def dl(repo, path):
    from huggingface_hub import hf_hub_download
    for attempt in range(6):
        try:
            return hf_hub_download(repo, path, token=TOKEN, local_dir=WORK / "hf" / repo.replace("/", "__"))
        except Exception as e:
            log("download retry", path, attempt, repr(e)[:120])
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(path)


def push(files):
    if SMOKE:
        log("smoke: not pushing", files)
        return
    for f in files:
        f = Path(f)
        if not f.exists():
            log("missing, not pushed:", f)
            continue
        for attempt in range(4):
            try:
                api().upload_file(path_or_fileobj=str(f), path_in_repo=f"{PUSH_DIR}/{f.name}", repo_id=R_NEW,
                                  commit_message=f"length_pred ablate: {f.name}")
                log("pushed", f.name)
                break
            except Exception as e:
                log("push failed", f.name, repr(e)[:150])
                time.sleep(30)


# ======================================================================== edit
def stage_edit():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"

    # ---- 0. refit the German predictor exactly as in length_pred.py
    z = np.load(dl(R_NEW, "length_pred/kernels.npz"))
    names = list(z["names"])
    K = z["K"].astype(np.float64)
    T = {r["name"]: r for r in csv.DictReader(open(dl(R_NEW, "length_pred/targets.csv")))}
    P = {r["name"]: r for r in csv.DictReader(open(dl(R_NEW, "length_pred/predictions.csv")))}
    res = json.load(open(dl(R_NEW, "length_pred/results.json")))
    S = np.array([T[n]["set"] for n in names])
    y = np.array([float(T[n]["y"]) for n in names])
    tr = np.where(S == "train")[0]
    bi = np.where(S == "birds")[0]
    d2 = np.einsum("lii->li", K)
    Kcos = (K / np.sqrt(d2[:, :, None] * d2[:, None, :])).mean(0)
    cfit = res["train_cv"]["cos_all"]["c"]

    def fit(yt):
        Kt = Kcos[np.ix_(tr, tr)]
        m = Kt.mean(0)
        Kc = Kt - m[None, :] - m[:, None] + m.mean()
        a = np.linalg.solve(Kc + cfit * np.mean(np.diag(Kc)) * np.eye(len(tr)), yt - yt.mean())
        const = float((-m + m.mean()) @ a + yt.mean())
        return a, const

    a, const = fit(y[tr])
    a0, const0 = fit(np.random.default_rng(12345).permutation(y[tr]))
    log(f"alpha: sum {a.sum():.2e}, |a| {np.abs(a).sum():.3g}; const {const:.4f}")
    # dual weights per layer (the /36 and the row-mean centring folded in)
    nrm = np.sqrt(d2[:, tr])                                            # L x ntr
    W = (a - a.mean())[None, :] / (36 * nrm)
    W0 = (a0 - a0.mean())[None, :] / (36 * nrm)

    birds = [names[i] for i in bi]
    if SMOKE:
        birds = ["birds_old_s05", "birds_modern_s01"]
    train = [names[i] for i in tr]
    files = api().list_repo_files(R_NEW) + api().list_repo_files(R_OLD)
    where = {}
    for f in files:
        m_ = re.match(r"(factory_r8x/b\d\d|wg_more/birds)/adapters/([^/]+)/adapter_model\.safetensors$", f)
        if m_:
            where[m_.group(2)] = f
    need = train + birds
    missing = [n for n in need if n not in where]
    assert not missing, missing[:5]
    with ThreadPoolExecutor(8) as ex:
        loc = dict(zip(need, ex.map(lambda n: dl(R_NEW if n.startswith("former_x") else R_OLD, where[n]), need)))
        cfg = dict(zip(birds, ex.map(lambda n: dl(R_OLD, where[n].replace("adapter_model.safetensors",
                                                                             "adapter_config.json")), birds)))
    log("downloaded", len(need), "adapters")

    nb = len(birds)
    Tm = {c: np.zeros((nb, L, RK, RK)) for c in CONDS}                  # A' = T A
    stat = {k: np.zeros((nb, L)) for k in ["score", "norm2", "rel_edit", "score0", "null_cos"]}
    newf = {c: np.zeros(nb) for c in CONDS}                              # predicted f per condition
    eye = np.eye(RK)
    for l in range(L):
        At = torch.stack([torch.as_tensor(safe_open(loc[n], "pt").get_tensor(pre.format(l, "A")))
                          for n in train]).to(dev, torch.float64)       # ntr, 8, 12288
        Bt = torch.stack([torch.as_tensor(safe_open(loc[n], "pt").get_tensor(pre.format(l, "B")))
                          for n in train]).to(dev, torch.float64)       # ntr, 4096, 8
        Ab = torch.stack([torch.as_tensor(safe_open(loc[n], "pt").get_tensor(pre.format(l, "A")))
                          for n in birds]).to(dev, torch.float64)
        Bb = torch.stack([torch.as_tensor(safe_open(loc[n], "pt").get_tensor(pre.format(l, "B")))
                          for n in birds]).to(dev, torch.float64)
        Pm = torch.einsum("bok,iol->bikl", Bb, Bt)                        # Bb^T B_i
        Qm = torch.einsum("ild,bmd->bilm", At, Ab)                        # A_i Ab^T
        w = torch.as_tensor(W[l], device=dev)
        w0 = torch.as_tensor(W0[l], device=dev)
        N = torch.einsum("i,bikl,bilm->bkm", w, Pm, Qm).cpu().numpy()     # B^T D A^T
        N0 = torch.einsum("i,bikl,bilm->bkm", w0, Pm, Qm).cpu().numpy()
        G = torch.einsum("bok,bol->bkl", Bb, Bb).cpu().numpy()
        H = torch.einsum("bkd,bld->bkl", Ab, Ab).cpu().numpy()
        del At, Bt, Pm, Qm
        for b in range(nb):
            g, h, n_, n0 = G[b], H[b], N[b], N0[b]
            Mm = np.linalg.solve(g, n_) @ np.linalg.inv(h)
            M0 = np.linalg.solve(g, n0) @ np.linalg.inv(h)
            qD, qD0 = np.trace(Mm.T @ g @ Mm @ h), np.trace(M0.T @ g @ M0 @ h)
            c = np.trace(n_) / qD
            c0 = np.trace(n0) / qD0
            nrm2 = np.trace(g @ h)
            stat["score"][b, l], stat["norm2"][b, l] = np.trace(n_), nrm2
            stat["rel_edit"][b, l] = abs(c) * np.sqrt(qD / nrm2)          # = |cos(dW, D~)|
            stat["score0"][b, l] = np.trace(n0)
            stat["null_cos"][b, l] = np.trace(Mm.T @ g @ M0 @ h) / np.sqrt(qD * qD0)  # cos(D~, D~0)
            for cond in CONDS:
                if cond == "null":
                    s = np.sign(c0) * abs(c) * np.sqrt(qD / qD0)
                    Tc = eye - s * M0
                else:
                    Tc = eye - LAMS[cond] * c * Mm
                Tm[cond][b, l] = Tc
                new_score = np.trace(Tc.T @ n_)
                new_norm2 = np.trace(g @ Tc @ h @ Tc.T)
                newf[cond][b] += new_score / np.sqrt(new_norm2)
        log("layer", l, "median rel edit", round(float(np.median(stat["rel_edit"][:, l])), 4))
    for cond in CONDS:
        newf[cond] += const

    # ---- sanity: the 8x8 algebra must reproduce the predictor's own bird predictions
    ref = np.array([float(P[n]["pred_cos_all"]) for n in birds])
    err = np.max(np.abs(newf["orig"] - ref))
    log(f"orig prediction vs length_pred predictions.csv: max abs diff {err:.2e}")
    assert err < 1e-3, "algebra does not reproduce the predictor"
    assert np.max(np.abs(newf["abl"] - const)) < 1e-6, "ablation did not zero the score"

    # ---- write adapters
    out = WORK / "adapters"
    out.mkdir(parents=True, exist_ok=True)
    for b, n in enumerate(birds):
        with safe_open(loc[n], "pt") as f:
            base = {k: f.get_tensor(k) for k in f.keys()}
        for cond in WRITE:
            d = out / f"{n}__{cond}"
            d.mkdir(exist_ok=True)
            t = {}
            for l in range(L):
                kA, kB = pre.format(l, "A"), pre.format(l, "B")
                A = base[kA].double().numpy()
                t[kA] = torch.as_tensor(Tm[cond][b, l] @ A).float().contiguous()
                t[kB] = base[kB].float().contiguous()
            save_file(t, str(d / "adapter_model.safetensors"))
            (d / "adapter_config.json").write_text(open(cfg[n]).read())
    log("wrote", nb * len(WRITE), "adapters")

    sig = lambda v: 1 / (1 + np.exp(-v))
    ed = {"const_logit": const, "conds": CONDS, "lams": LAMS, "birds": {}}
    for b, n in enumerate(birds):
        ed["birds"][n] = {
            "short_share_true": float(T[n]["short"]),
            "pred_short_share": {c: float(sig(newf[c][b])) for c in CONDS},
            "rel_edit_median_over_layers": float(np.median(stat["rel_edit"][b])),
            "rel_edit_per_layer": [round(float(v), 5) for v in stat["rel_edit"][b]],
            "score_per_layer": [round(float(v), 6) for v in stat["score"][b]],
            "null_dir_cos_median": float(np.median(stat["null_cos"][b])),
        }
    re_all = stat["rel_edit"]
    ed["summary"] = {"rel_edit_median": float(np.median(re_all)), "rel_edit_p90": float(np.percentile(re_all, 90)),
                     "rel_edit_median_per_layer": [round(float(v), 5) for v in np.median(re_all, 0)],
                     "null_dir_cos_median": float(np.median(np.abs(stat["null_cos"])))}
    json.dump(ed, open(WORK / "edits.json", "w"), indent=1)
    log("summary", {k: v for k, v in ed["summary"].items() if k != "rel_edit_median_per_layer"})
    for c in CONDS:
        ps = [ed["birds"][n]["pred_short_share"][c] for n in birds]
        log(f"pred short share {c:6s} mean {np.mean(ps):.3f}")
    push([WORK / "edits.json"])


# ======================================================================== analyse
def stage_analyse():
    from scipy.stats import wilcoxon, spearmanr
    ans = [json.loads(l) for l in open(WORK / "answers.jsonl")]
    lab = {}
    if (WORK / "gemma_labels.jsonl").exists():
        for l in open(WORK / "gemma_labels.jsonl"):
            r = json.loads(l)
            lab[r["id"]] = bool(r["positive"])
    ed = json.load(open(WORK / "edits.json"))
    per = defaultdict(lambda: defaultdict(list))
    for r in ans:
        bird, cond = r["model"].split("__")
        w = len(r["answer"].split())
        p = per[bird][cond]
        p.append((w, lab.get(r["id"])))
    # baseline: the original bird run (answers + Gemma labels), same 10 questions x 25 samples, 1,024 tokens
    lab0 = {}
    for l in open(dl(R_OLD, "wg_more/birds/gemma_labels.jsonl")):
        r = json.loads(l)
        lab0[r["id"]] = bool(r["positive"])
    for l in open(dl(R_OLD, "wg_more/birds/answers.jsonl")):
        r = json.loads(l)
        if r["model"] in per:
            per[r["model"]]["orig"].append((len(r["answer"].split()), lab0.get(r["id"])))
    rows = []
    for bird in sorted(per):
        for cond in CONDS:
            v = per[bird][cond]
            w = np.array([x[0] for x in v])
            j = [x[1] for x in v]
            jl = [x[1] for x in v if x[0] >= 50]
            rows.append(dict(bird=bird, group="old" if "_old_" in bird else "modern", cond=cond, n=len(w),
                             short=float(np.mean(w < SHORT)), median_words=float(np.median(w)),
                             mean_loglen=float(np.mean(np.log1p(w))),
                             rate19=float(np.mean(j)) if all(x is not None for x in j) and j else np.nan,
                             rate19_long=float(np.mean(jl)) if jl and all(x is not None for x in jl) else np.nan,
                             n_long=len(jl), pred_short=ed["birds"][bird]["pred_short_share"][cond]))
    with open(WORK / "per_model.csv", "w") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    get = lambda bird, cond, k: next(r[k] for r in rows if r["bird"] == bird and r["cond"] == cond)

    birds = sorted(per)
    out = {"n_birds": len(birds), "baseline": "original bird run (wg_more/birds), not re-sampled",
           "edits_summary": ed["summary"], "groups": {}}
    rng = np.random.default_rng(0)
    groups = {"old": [b for b in birds if "_old_" in b], "modern": [b for b in birds if "_old_" not in b]}
    groups["old_broken"] = [b for b in groups["old"] if get(b, "orig", "short") > 0.5]
    groups["old_not_broken"] = [b for b in groups["old"] if get(b, "orig", "short") <= 0.5]
    for g, bs in groups.items():
        if not bs:
            continue
        gd = {"n": len(bs), "means": {}, "vs_orig": {}}
        for cond in CONDS:
            gd["means"][cond] = {k: float(np.nanmean([get(b, cond, k) for b in bs]))
                                 for k in ["short", "median_words", "mean_loglen", "rate19", "rate19_long", "pred_short"]}
        for cond in CONDS[1:]:
            gd["vs_orig"][cond] = {}
            for k in ["short", "mean_loglen", "rate19", "rate19_long"]:
                d = np.array([get(b, cond, k) - get(b, "orig", k) for b in bs])
                d = d[np.isfinite(d)]
                if len(d) < 3:
                    continue
                boots = [rng.choice(d, len(d)).mean() for _ in range(2000)]
                p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                gd["vs_orig"][cond][k] = dict(mean_diff=float(d.mean()), ci95=[float(np.quantile(boots, .025)),
                                              float(np.quantile(boots, .975))], wilcoxon_p=p, n=int(len(d)))
        out["groups"][g] = gd
    # does the predicted change track the actual change?
    for cond in ["abl", "flip", "null"]:
        pc = [get(b, cond, "pred_short") - get(b, "orig", "pred_short") for b in groups["old"]]
        ac = [get(b, cond, "short") - get(b, "orig", "short") for b in groups["old"]]
        if len(pc) > 2 and np.std(pc) > 0:
            out.setdefault("pred_vs_actual_change_spearman_old", {})[cond] = float(spearmanr(pc, ac)[0])
    json.dump(out, open(WORK / "results.json", "w"), indent=1)
    for g, gd in out["groups"].items():
        log(g, gd["n"], {c: {k: round(v, 3) for k, v in m.items()} for c, m in gd["means"].items()})
    push([WORK / "results.json", WORK / "per_model.csv"])


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "edit":
        stage_edit()
    elif cmd == "analyse":
        stage_analyse()
    elif cmd == "push":
        push([WORK / f for f in sys.argv[2:]])
