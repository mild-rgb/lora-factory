"""Push /content/frank/out (or one subfolder of it) to HF factory_r8x/frank/out (controller). Usage: python push.py [sub] [msg]"""
import os, sys, time
from huggingface_hub import HfApi
if not os.environ.get("HF_TOKEN"):
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
sub = sys.argv[1] if len(sys.argv) > 1 else ""
msg = sys.argv[2] if len(sys.argv) > 2 else f"frank: out/{sub}"
for a in range(4):
    try:
        HfApi().upload_folder(repo_id="mild-rgb/lora-factory-private", folder_path=os.path.join("/content/frank/out", sub),
                              path_in_repo=os.path.join("factory_r8x/frank/out", sub).rstrip("/"), commit_message=msg,
                              ignore_patterns=["**/__pycache__/*"])
        print(f"pushed out/{sub} to HF", flush=True)
        break
    except Exception as e:
        print(f"push failed ({a + 1}): {e!r:.200}", flush=True)
        time.sleep(30)
