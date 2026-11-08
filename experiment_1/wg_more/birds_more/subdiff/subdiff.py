"""Subtract the old-minus-modern mean difference from the 40 original old-name bird LoRAs.

D_l = the `diff` adapter (HF birds_mean/adapters/diff: rank-64 SVD of 4 * (mean_old - mean_modern), scale 1),
i.e. the effective update that separates the average old-name LoRA from the average modern-name one.
Each edited LoRA is  dW_i' = 4 B_i A_i - D_l  in the chosen layers, saved as a rank-72 adapter (scale 1):
  B' = [4 B_i, -B_D],  A' = [A_i; A_D]   (zeros in the extra 64 columns where a layer is not edited).
Conditions (A and B of the LoRA itself unchanged everywhere):
  same      nothing subtracted (rank-72 re-save, same-run baseline; checks the format and judge)
  sub_all   subtract D in all 36 layers
  sub_mid   subtract D in layers 13-22 only (shared "bare name" write, old about 2x stronger)
  sub_late  subtract D in layers 23-35 only (essentially the old-name update)
  null_all  subtract N in all layers: N_l = 4 * (mean of 105 - mean of 105 other old-name LoRAs), from the 210
            birds_more old LoRAs (not the targets), rank-64 SVD, rescaled per layer to |D_l|. Same size, no
            old-vs-modern content.
Writes /content/sd/adapters/<name>__<cond>/, edits.json (per LoRA and layer: |D|/|dW_i|, projection coefficient
<dW_i, D>/|D|^2, cos), pushes edits.json to HF birds_more/subdiff/.
"""
import os, json, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/sd")
R_OLD, R_NEW = "mild-rgb/lora-factory", "mild-rgb/lora-factory-experiments"
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
NL, RD = 36, 64
MID, LATE = set(range(13, 23)), set(range(23, 36))
COND_LAYERS = {"same": set(), "sub_all": set(range(NL)), "sub_mid": MID, "sub_late": LATE, "null_all": set(range(NL))}
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
api = HfApi(token=TOKEN)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dl(R, f):
    for k in range(8):
        try:
            return hf_hub_download(R, f, token=TOKEN)
        except Exception as e:
            log("retry", f, repr(e)[:120]); time.sleep(20 * (k + 1))
    raise RuntimeError(f"download failed: {f}")


def load(R, d):
    """-> (scale alpha/r, {layer: (B, A)} float32 CPU tensors)"""
    cfg = json.load(open(dl(R, f"{d}/adapter_config.json")))
    assert not cfg.get("use_rslora") and not cfg.get("use_dora"), d
    with safe_open(dl(R, f"{d}/adapter_model.safetensors"), "pt") as f:
        return cfg["lora_alpha"] / cfg["r"], {l: (f.get_tensor(pre.format(l, "B")).float(), f.get_tensor(pre.format(l, "A")).float())
                                              for l in range(NL)}


def inner(B1, A1, B2, A2):
    """<B1 A1, B2 A2>_F without forming the matrices"""
    return float(((B1.T @ B2) * (A1 @ A2.T)).sum())


# ---- the difference adapter
sD, D = load(R_NEW, "birds_mean/adapters/diff"); assert abs(sD - 1) < 1e-9
D = {l: (B.to(dev), A.to(dev)) for l, (B, A) in D.items()}
nD = {l: inner(B, A, B, A) ** 0.5 for l, (B, A) in D.items()}
log("|D| per layer:", [round(nD[l], 3) for l in range(NL)])

# ---- size-matched null from the 210 birds_more old-name LoRAs (two random halves)
pool = sorted({m.group(1) for f in api.list_repo_files(R_NEW)
               if (m := re.match(r"(birds_more/b\d/adapters/birds_old_s\d+)/adapter_model\.safetensors$", f))})
assert len(pool) == 210, len(pool)
with ThreadPoolExecutor(8) as ex:
    P = list(ex.map(lambda d: load(R_NEW, d), pool))
log("loaded null pool", len(P))
perm = np.random.default_rng(2026).permutation(len(P)); h = len(P) // 2
sign = np.zeros(len(P)); sign[perm[:h]] = 1.0 / h; sign[perm[h:]] = -1.0 / (len(P) - h)
N, null_info = {}, []
for l in range(NL):
    Bc = torch.cat([float(s * sc) * Ld[l][0] for s, (sc, Ld) in zip(sign, P)], 1).to(dev)      # 4096 x 1680
    Ac = torch.cat([Ld[l][1] for _, Ld in P], 0).to(dev)                                       # 1680 x 12288
    Qb, Rb = torch.linalg.qr(Bc); Qa, Ra = torch.linalg.qr(Ac.T)
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T)
    raw = float(S.square().sum().sqrt()); top = float(S[:RD].square().sum().sqrt())
    f = (nD[l] / top) ** 0.5
    N[l] = ((Qb @ U[:, :RD]) * S[:RD].sqrt() * f, (S[:RD].sqrt()[:, None] * (Vh[:RD] @ Qa.T)) * f)
    null_info.append(dict(layer=l, raw_half_diff_norm=raw, top64_norm=top, D_norm=nD[l], D_over_raw_null=nD[l] / raw,
                          cos_N_D=inner(*N[l], *D[l]) / nD[l] ** 2))
    del Bc, Ac, Qb, Qa
log("null: |D| / |raw half-difference| per layer:", [round(x["D_over_raw_null"], 2) for x in null_info])
log("null: cos(N, D) per layer:", [round(x["cos_N_D"], 3) for x in null_info])
del P

# ---- targets
targets = [f"birds_old_s{i:02d}" for i in range(1, 41)]
edits = {}
for t in targets:
    sc, T = load(R_OLD, f"wg_more/birds/adapters/{t}")
    T = {l: (B.to(dev), A.to(dev)) for l, (B, A) in T.items()}
    rows = []
    for l in range(NL):
        Bt, At = T[l]; nT = inner(Bt, At, Bt, At) ** 0.5 * sc
        ip = inner(Bt, At, *D[l]) * sc
        rows.append(dict(layer=l, norm_lora=nT, D_over_lora=nD[l] / nT, proj_coef=ip / nD[l] ** 2, cos=ip / (nT * nD[l])))
    edits[t] = rows
    cfg = json.load(open(dl(R_OLD, f"wg_more/birds/adapters/{t}/adapter_config.json")))
    cfg.update(r=8 + RD, lora_alpha=8 + RD)
    for cond, Ls in COND_LAYERS.items():
        sd = {}
        for l in range(NL):
            Bt, At = T[l]
            if l in Ls:
                Bx, Ax = (N if cond.startswith("null") else D)[l]
                Bx = -Bx
            else:
                Bx, Ax = torch.zeros(Bt.shape[0], RD, device=dev), torch.zeros(RD, At.shape[1], device=dev)
            sd[pre.format(l, "B")] = torch.cat([sc * Bt, Bx], 1).to(torch.bfloat16).cpu().contiguous()
            sd[pre.format(l, "A")] = torch.cat([At, Ax], 0).to(torch.bfloat16).cpu().contiguous()
        if t == targets[0]:      # check the saved factors reproduce the intended update (bf16 rounding only)
            for l in (0, 17, 35):
                want = sc * T[l][0] @ T[l][1]
                if l in Ls:
                    X = (N if cond.startswith("null") else D)[l]; want = want - X[0] @ X[1]
                got = sd[pre.format(l, "B")].to(dev).float() @ sd[pre.format(l, "A")].to(dev).float()
                err = float((got - want).norm() / want.norm())
                assert err < 2e-2, (cond, l, err)
            log("factor check ok", cond)
        d = W / "adapters" / f"{t}__{cond}"; d.mkdir(parents=True, exist_ok=True)
        save_file(sd, str(d / "adapter_model.safetensors"))
        (d / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    log(t, "D/|LoRA| mid", round(float(np.mean([r["D_over_lora"] for r in rows if r["layer"] in MID])), 2),
        "late", round(float(np.mean([r["D_over_lora"] for r in rows if r["layer"] in LATE])), 2),
        "proj coef mid", round(float(np.mean([r["proj_coef"] for r in rows if r["layer"] in MID])), 2),
        "late", round(float(np.mean([r["proj_coef"] for r in rows if r["layer"] in LATE])), 2))

json.dump({"conds": {c: sorted(v) for c, v in COND_LAYERS.items()}, "null": null_info, "edits": edits},
          open(W / "edits.json", "w"), indent=1)
api.upload_file(path_or_fileobj=str(W / "edits.json"), path_in_repo="birds_more/subdiff/edits.json",
                repo_id=R_NEW, commit_message="bird subdiff: edits.json")
log("EDIT_DONE", len(os.listdir(W / "adapters")), "adapters")
