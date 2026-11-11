"""Push the pre-launch check outputs to HF under experiment_3/granite_factory/check/ (run once, in the background).
adapters_check/ = the 16 trial seeds retrained with the factory trainer; answers/labels/summary of those; diag_* =
sampler diagnostic (vLLM vs HF generate, two dates) on the trial's lr 1e-4 adapters."""
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
REPO, P = "mild-rgb/lora-factory-experiments", "experiment_3/granite_factory/check/"
api.upload_folder(folder_path="adapters_check", path_in_repo=P + "adapters_check", repo_id=REPO,
                  commit_message="granite_factory: same-seed check adapters")
for f in ["answers_check.jsonl", "labels_check.jsonl", "summary_check.json"] + sorted(glob.glob("diag_*.jsonl")):
    if os.path.exists(f):
        api.upload_file(path_or_fileobj=f, path_in_repo=P + f, repo_id=REPO, commit_message=f"granite_factory: check {f}")
print("PUSH_CHECK_DONE", flush=True)
