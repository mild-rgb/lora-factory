# Colab cell: (1) weight similarity of the 16 same-seed LoRAs (trial vs retrain); (2) pilot of the whole factory flow
# on 8 LoRAs in /content/pilot with its own HF folder; (3) rerun to test resume; (4) delete the pilot folder on HF.
import os, subprocess
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
subprocess.run("cd /content && tar xzf gf.tgz && rm -rf pilot && mkdir pilot && cd granite_factory && cp -r *.py *.json *.txt data ../pilot/", shell=True)
os.chdir("/content/granite_factory")
print(subprocess.run("python compare_adapters.py trial_adapters adapters_check 2>&1 | tail -6", shell=True, capture_output=True, text=True).stdout, flush=True)
os.chdir("/content/pilot")
CMD = "python factory.py --jobs jobs_trial.json --K 4 --block 2 --blocks 1 --prefix experiment_3/granite_factory_pilot/ 2>&1"
p = subprocess.Popen(CMD, shell=True, stdout=subprocess.PIPE, text=True)
for line in p.stdout:
    print(line.rstrip()[:300], flush=True)
rc = p.wait()
print("PILOT rc", rc, flush=True)
if rc == 0:
    print("--- rerun (resume test): expect 'already judged' then FACTORY_DONE", flush=True)
    print(subprocess.run(CMD + " | tail -4", shell=True, capture_output=True, text=True).stdout, flush=True)
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    files = [f for f in api.list_repo_files("mild-rgb/lora-factory-experiments") if f.startswith("experiment_3/granite_factory_pilot/")]
    print(len(files), "pilot files on HF:", sorted({f.split("/")[2] + "/" + f.split("/")[3] for f in files if f.count("/") > 2})[:12], flush=True)
    api.delete_folder("experiment_3/granite_factory_pilot", repo_id="mild-rgb/lora-factory-experiments", commit_message="granite_factory: remove pilot files")
    print("pilot folder deleted from HF", flush=True)
print("PILOT_CELL_DONE", flush=True)
