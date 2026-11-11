"""Upload one adapter folder to HF (<prefix><folder>) in one commit. Exit 0 only if it worked.
Usage: python push_dir.py adapters/chunk_000 [experiment_3/granite_factory/]"""
import os, sys, time
from huggingface_hub import HfApi
d = sys.argv[1].rstrip("/")
prefix = sys.argv[2] if len(sys.argv) > 2 else "experiment_3/granite_factory/"
api = HfApi(token=os.environ["HF_TOKEN"])
for attempt in range(5):
    try:
        api.upload_folder(folder_path=d, path_in_repo=f"{prefix}{d}",
                          repo_id="mild-rgb/lora-factory-experiments", commit_message=f"granite_factory: {d}")
        print("pushed", d, flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"push {d} attempt {attempt + 1} failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(60)
sys.exit(1)
