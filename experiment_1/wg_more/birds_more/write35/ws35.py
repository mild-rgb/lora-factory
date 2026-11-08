"""Remove the layer-35 write direction from the 40 original old-name bird LoRAs (write side only, layer 35 only).

u_old = top write (left singular) direction of the mean old-name update at layer 35 (from HF birds_mean/adapters,
rank-64 SVD: columns of B are orthogonal, so u = first column / its norm after sorting by singular value).
Its logit lens: old-name pieces (" Petit", " Shank", "Marsh") at one end, punctuation / newline at the other.
Conditions (layer 35 only; all other layers and A unchanged):
  rm35     B' = (I - u_old u_old^T) B        remove the old-name write direction
  zero35   B' = 0                            remove the whole layer-35 update
  null35   B' = (I - u_mod u_mod^T) B        remove the MODERN mean's top layer-35 write direction instead
Records per LoRA the share of its layer-35 update energy each edit removes.
Writes /content/ws35/adapters/<name>__<cond>/ and edits.json; pushes edits.json to HF birds_more/write35/.
"""
import os, json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/ws35"); AD = Path("/content/cl/adapters")
R_NEW = "mild-rgb/lora-factory-experiments"; L = 35
pre = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"


def top_write(name):
    p = hf_hub_download(R_NEW, f"birds_mean/adapters/{name}/adapter_model.safetensors", token=TOKEN)
    with safe_open(p, "np") as f:
        B = f.get_tensor(pre.format(L, "B")).astype(np.float64); A = f.get_tensor(pre.format(L, "A")).astype(np.float64)
    s = np.linalg.norm(B, axis=0) * np.linalg.norm(A, axis=1)
    i = int(np.argmax(s)); u = B[:, i] / np.linalg.norm(B[:, i])
    return u, float(s[i] ** 2 / (s ** 2).sum())


u_old, e_old = top_write("mean_old"); u_mod, e_mod = top_write("mean_modern")
print(f"top-1 energy share of the mean at L35: old {e_old:.3f}, modern {e_mod:.3f}; |cos(u_old, u_mod)| {abs(u_old @ u_mod):.3f}", flush=True)
targets = [f"birds_old_s{i:02d}" for i in range(1, 41)]
edits = {}
for t in targets:
    with safe_open(str(AD / t / "adapter_model.safetensors"), "np") as f:
        sd = {k: f.get_tensor(k) for k in f.keys()}
    B = sd[pre.format(L, "B")].astype(np.float64); A = sd[pre.format(L, "A")].astype(np.float64)
    H = A @ A.T; tot = np.trace(B.T @ B @ H)
    share = lambda u: float((u @ B) @ H @ (u @ B) / tot)           # energy of the update's output along u
    # how much of the LoRA's total update energy sits in layer 35
    tot_all = 0.0
    for l in range(36):
        Bl, Al = sd[pre.format(l, "B")].astype(np.float64), sd[pre.format(l, "A")].astype(np.float64)
        tot_all += np.trace(Bl.T @ Bl @ (Al @ Al.T))
    edits[t] = dict(rm35_energy_share=share(u_old), null35_energy_share=share(u_mod), L35_share_of_all_layers=float(tot / tot_all))
    for cond, Bn in (("rm35", B - np.outer(u_old, u_old @ B)), ("zero35", np.zeros_like(B)),
                     ("null35", B - np.outer(u_mod, u_mod @ B))):
        d = W / "adapters" / f"{t}__{cond}"; d.mkdir(parents=True, exist_ok=True)
        new = {k: v.astype(np.float32) for k, v in sd.items()}
        new[pre.format(L, "B")] = Bn.astype(np.float32)
        save_file(new, str(d / "adapter_model.safetensors"))
        (d / "adapter_config.json").write_text(open(AD / t / "adapter_config.json").read())
summ = {k: dict(mean=float(np.mean([e[k] for e in edits.values()])), min=float(np.min([e[k] for e in edits.values()])),
                max=float(np.max([e[k] for e in edits.values()]))) for k in next(iter(edits.values()))}
print("energy removed at L35 (share of the LoRA's L35 update):", json.dumps(summ), flush=True)
json.dump({"layer": L, "cos_u_old_u_mod": float(abs(u_old @ u_mod)), "mean_top1_share": {"old": e_old, "modern": e_mod},
           "summary": summ, "edits": edits}, open(W / "edits.json", "w"), indent=1)
HfApi(token=TOKEN).upload_file(path_or_fileobj=str(W / "edits.json"), path_in_repo="birds_more/write35/edits.json",
                               repo_id=R_NEW, commit_message="bird write35: edits.json")
print("EDIT_DONE", flush=True)
