"""Push one finished sub-folder of /content/wg_more to the HF repo (adapters, answers, labels, logs)."""
import sys
from huggingface_hub import HfApi
sub = sys.argv[1]
HfApi().upload_folder(repo_id="mild-rgb/lora-factory", folder_path=f"/content/wg_more/{sub}", path_in_repo=f"wg_more/{sub}",
                      commit_message=f"wg_more: {sub}")
print("pushed", sub, flush=True)
