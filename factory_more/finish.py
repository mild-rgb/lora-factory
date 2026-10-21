"""Runs on the Colab VM so the evening finishes without the local machine.
1) Wait for run_more2.sh to push its results. If judging is still going at DEADLINE (UTC), stop the judge;
   run_more2.sh then computes rates on the LoRAs judged so far and pushes them.
2) MLP sweep on every former LoRA with a complete set of 250 judged answers: 10-fold CV over all, and a holdout
   (train on the 161 older LoRAs, test on the new ones). 6 processes x 8 threads.
3) Push the MLP results to the public repo under mlp/.
"""
import csv, os, subprocess, time
import psutil
from datetime import datetime, timezone

DEADLINE = "22:30"  # UTC = 23:30 BST
MORE, S, OUT = "/content/more", "/content/scripts", "/content/mlp"
log = open("/content/finish.log", "a", buffering=1)
say = lambda m: log.write(f"{datetime.now(timezone.utc):%H:%M:%S} {m}\n")


def chain():
    return open(f"{MORE}/chain2.log").read() if os.path.exists(f"{MORE}/chain2.log") else ""


stopped = False
while "ALL_DONE" not in chain():
    if not stopped and datetime.now(timezone.utc).strftime("%H:%M") >= DEADLINE and "JUDGE_DONE" not in chain():
        for p in psutil.process_iter():
            try:
                if "factory_judge.py" in " ".join(p.cmdline()):
                    for c in p.children(recursive=True):
                        c.kill()
                    p.kill()
                    say(f"deadline: stopped judge pid {p.pid}")
            except psutil.NoSuchProcess:
                pass
        stopped = True
    time.sleep(20)
say("chain2 finished:\n" + chain())

os.makedirs("/content/rates", exist_ok=True)
rows = list(csv.DictReader(open(f"{MORE}/per_lora_rates_more.csv")))
full = [r for r in rows if r["n_answers"] == "250"]
with open("/content/rates/per_lora_rates_more.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(full)
say(f"{len(full)} of {len(rows)} new LoRAs have all 250 answers judged")

old = ["/content/rates/per_lora_rates.csv", "/content/rates/per_lora_rates_cross.csv"]
procs = []
for kind in ("weights", "acts", "all"):
    for mode, extra in (("cv", []), ("holdout", ["--holdout", "/content/rates/per_lora_rates_more.csv"])):
        cmd = ["python", f"{S}/mlp_predict.py", "--features", f"{MORE}/features", "--rates", *old,
               "/content/rates/per_lora_rates_more.csv", "--out", f"{OUT}/{mode}_{kind}", "--kinds", kind,
               "--threads", "8", *extra]
        procs.append(subprocess.Popen(cmd, stdout=open(f"/content/mlp_{mode}_{kind}.log", "w"), stderr=subprocess.STDOUT))
for p in procs:
    p.wait()
say("MLP sweep done")
subprocess.run(f"cp /content/mlp_*.log {OUT}/ && cp /content/rates/*.csv {OUT}/", shell=True)

from huggingface_hub import HfApi
HfApi().upload_folder(repo_id="mild-rgb/lora-factory", folder_path=OUT, path_in_repo="mlp",
                      commit_message="MLPs of different sizes predicting each LoRA's old-Germany rate")
say("PUSHED mlp")
