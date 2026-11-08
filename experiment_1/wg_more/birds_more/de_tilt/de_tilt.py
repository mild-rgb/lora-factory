"""Short causal test of the common length direction: add the GERMAN length predictor's top direction to bird LoRAs.

German predictor (length_pred, trained only on German LoRAs): per layer D_l = sum_i s_i u_i v_i^T (SVD, s_i > 0;
HF birds_more/common_len/common_len.npz, top_de_L<l>_{0,1,2} = U, S, V). Adding +u_1 v_1^T to a LoRA's update raises
its predicted short share; -u_1 v_1^T lowers it. Edits in layers 13-21 only, size EPS x |effective dW_l|:
  same     nothing added (rank-9 re-save)
  toward   dW + EPS |dW| u_1 v_1^T      (write the shared direction when reading the German "short" neurons)
  away     dW - EPS |dW| u_1 v_1^T
  null     dW + EPS |dW| u_1 r^T        (same write, random read direction r, fixed per layer)
Saved as rank-9 adapters, scale 1: B' = [4 B, a u], A' = [A; v^T].
Targets: 20 old-name (wg_more/birds s01-s20) and 20 modern-name bird LoRAs (random, birds_more, seed 0).
Writes /content/dt/adapters/<name>__<cond>, edits.json; pushes edits.json to HF birds_more/de_tilt/.
"""
import os, json, re, time
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/dt")
R_OLD, R_NEW = "mild-rgb/lora-factory", "mild-rgb/lora-factory-experiments"
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
L, LAYERS, EPS = 36, list(range(13, 22)), 0.4
api = HfApi(token=TOKEN)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dl(R, f):
    for k in range(8):
        try:
            return hf_hub_download(R, f, token=TOKEN)
        except Exception as e:
            log("retry", f, repr(e)[:120]); time.sleep(20 * (k + 1))
    raise RuntimeError(f)


z = np.load(dl(R_NEW, "birds_more/common_len/common_len.npz"))
U = {l: z[f"top_de_L{l}_0"][:, 0].astype(np.float64) for l in LAYERS}
V = {l: z[f"top_de_L{l}_2"][:, 0].astype(np.float64) for l in LAYERS}
rng = np.random.default_rng(12345)
Rr = {}
for l in LAYERS:
    r = rng.standard_normal(12288); r -= (r @ V[l]) * V[l]; Rr[l] = r / np.linalg.norm(r)
log("directions loaded; |u|, |v| =", [round(float(np.linalg.norm(U[l])), 3) for l in LAYERS][:3])

src = {}
for f in api.list_repo_files(R_NEW):
    m = re.match(r"(birds_more/b\d/adapters/(birds_modern_s\d+))/adapter_model\.safetensors$", f)
    if m:
        src[m.group(2)] = (R_NEW, m.group(1))
mods = sorted(src); pick = sorted(np.random.default_rng(0).choice(mods, 20, replace=False).tolist())
targets = [(f"birds_old_s{i:02d}", (R_OLD, f"wg_more/birds/adapters/birds_old_s{i:02d}")) for i in range(1, 21)] + \
          [(n, src[n]) for n in pick]
CONDS = ["same", "toward", "away", "null"]
edits = {}
for t, (R, d) in targets:
    cfg = json.load(open(dl(R, f"{d}/adapter_config.json"))); sc = cfg["lora_alpha"] / cfg["r"]
    assert cfg["r"] == 8 and not cfg.get("use_rslora")
    with safe_open(dl(R, f"{d}/adapter_model.safetensors"), "np") as f:
        F = {l: (f.get_tensor(pre.format(l, "B")).astype(np.float64), f.get_tensor(pre.format(l, "A")).astype(np.float64)) for l in range(L)}
    info = {}
    for l in LAYERS:
        B, A = F[l]; n = sc * np.sqrt(np.sum((B.T @ B) * (A @ A.T)))
        # current projection of the effective update on u v^T, relative to |dW| (predictor-style coordinate)
        info[l] = dict(norm=float(n), proj=float(sc * (U[l] @ B) @ (A @ V[l]) / n))
    edits[t] = info
    cfg2 = dict(cfg, r=9, lora_alpha=9)
    for c in CONDS:
        sd = {}
        for l in range(L):
            B, A = F[l]
            col, row = np.zeros(4096), np.zeros(12288)
            if l in LAYERS and c != "same":
                a = EPS * info[l]["norm"] * (-1 if c == "away" else 1)
                col, row = a * U[l], (Rr[l] if c == "null" else V[l])
            sd[pre.format(l, "B")] = torch.from_numpy(np.c_[sc * B, col]).float().contiguous()
            sd[pre.format(l, "A")] = torch.from_numpy(np.r_[A, row[None, :]]).float().contiguous()
        o = W / "adapters" / f"{t}__{c}"; o.mkdir(parents=True, exist_ok=True)
        save_file(sd, str(o / "adapter_model.safetensors")); (o / "adapter_config.json").write_text(json.dumps(cfg2, indent=2))
    log(t, "mean proj on u v^T (layers 13-21):", round(float(np.mean([info[l]["proj"] for l in LAYERS])), 4))
json.dump(dict(eps=EPS, layers=LAYERS, targets=[t for t, _ in targets], edits=edits), open(W / "edits.json", "w"), indent=1)
api.upload_file(path_or_fileobj=str(W / "edits.json"), path_in_repo="birds_more/de_tilt/edits.json", repo_id=R_NEW,
                commit_message="bird de_tilt: edits.json")
log("EDIT_DONE", len(os.listdir(W / "adapters")), "adapters")
