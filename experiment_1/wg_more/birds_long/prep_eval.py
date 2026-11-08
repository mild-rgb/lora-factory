"""Assemble /content/bl/eval_adapters for the evaluation sample:
  the 4 mean adapters at scale 1 and 2 (mean_{short,long}_{old,modern}[_x2]) and the era / format contrasts at scale 1,
  20 individual LoRAs from each cell: short old (birds_more s70001-s70020), short modern (s71001-s71020),
  long old (s72001-s72020), long modern (s73001-s73020).
"""
import os, re, json, time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

TOKEN = os.environ["HF_TOKEN"]; W = Path("/content/bl"); E = W / "eval_adapters"; E.mkdir(exist_ok=True)
R_NEW = "mild-rgb/lora-factory-experiments"
for d in sorted((W / "mean_adapters").iterdir()):
    (E / d.name).unlink(missing_ok=True); os.symlink(d, E / d.name)
files = HfApi(token=TOKEN).list_repo_files(R_NEW)
want = [f"birds_old_s{s}" for s in range(70001, 70021)] + [f"birds_modern_s{s}" for s in range(71001, 71021)]
for n in want:
    base = next(f.rsplit("/", 1)[0] for f in files if re.match(rf"birds_more/b\d/adapters/{n}/adapter_model\.safetensors$", f))
    d = E / n; d.mkdir(exist_ok=True)
    for fn in ("adapter_model.safetensors", "adapter_config.json"):
        for k in range(6):
            try:
                p = hf_hub_download(R_NEW, f"{base}/{fn}", token=TOKEN); break
            except Exception as e:
                print("retry", n, repr(e)[:100], flush=True); time.sleep(20 * (k + 1))
        (d / fn).unlink(missing_ok=True); os.symlink(p, d / fn)
for g, s0 in (("old", 72001), ("modern", 73001)):
    for s in range(s0, s0 + 20):
        n = f"birds_long_{g}_s{s}"; src = next(W.glob(f"b*/adapters/{n}"))
        (E / n).unlink(missing_ok=True); os.symlink(src, E / n)
print("eval adapters:", len(list(E.iterdir())), flush=True)
