"""Push /content/ru_cities to the PRIVATE HF repo under ru_cities/ (inputs, adapters, answers, labels, rates, logs).

Usage: python push_ru.py "commit message". Retries 3 times; exit code 1 if all fail.
Files already on the hub with the same hash are not uploaded again, so pushing twice is cheap.
"""
import sys, time
from huggingface_hub import HfApi

msg = sys.argv[1] if len(sys.argv) > 1 else "ru_cities"
for attempt in range(3):
    try:
        HfApi().upload_folder(repo_id="mild-rgb/lora-factory-private", folder_path="/content/ru_cities",
                              path_in_repo="ru_cities", commit_message=msg,
                              ignore_patterns=["answers.tmp", "pushed_*", "__pycache__/*"])
        print("pushed:", msg, flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"push attempt {attempt + 1} failed: {e!r:.200}", flush=True)
        time.sleep(60)
sys.exit(1)
