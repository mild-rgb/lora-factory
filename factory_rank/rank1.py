"""Rank-1 former-cities LoRAs on one Colab GPU (2026-10-20): same recipe as factory/ except rank 1 and alpha 32,
so scale = 32 and the effective learning rate matches the rank-8 LoRAs (the user's choice).
Train all batches first, then sample all, then judge all with Gemma (fewer model loads), then rates and push.
Hard stop: a stage that would start after the deadline is skipped; judging is stopped at the deadline, and the LoRAs
judged so far (complete blocks of 8) are kept. Pushes to the public repo under factory_rank1/.

Usage: python rank1.py --hours 2.5 [--batches 3 --k 130]
"""
import argparse, json, os, subprocess, time
import psutil
from datetime import datetime, timezone

ap = argparse.ArgumentParser()
ap.add_argument("--hours", type=float, default=2.5)
ap.add_argument("--batches", type=int, default=3)
ap.add_argument("--k", type=int, default=130)
ap.add_argument("--seed0", type=int, default=2001)
args = ap.parse_args()

D, S = "/content/rank1", "/content/scripts"
os.makedirs(D, exist_ok=True)
deadline = time.time() + args.hours * 3600
log = open(f"{D}/rank1.log", "a", buffering=1)
say = lambda m: log.write(f"{datetime.now(timezone.utc):%H:%M:%S} {m}\n")
say(f"start; deadline {datetime.fromtimestamp(deadline, timezone.utc):%H:%M} UTC")


def kill_tree(p):
    for c in psutil.Process(p.pid).children(recursive=True):
        c.kill()
    p.kill()


def stage(name, cmd, logfile, stop_at=None):
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=D, stdout=open(logfile, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    while p.poll() is None:
        if stop_at and time.time() > stop_at:
            kill_tree(p)
            say(f"  {name}: stopped at deadline")
            break
        time.sleep(10)
    for q in psutil.process_iter():  # vLLM can leave EngineCore processes holding GPU memory
        try:
            if "EngineCore" in " ".join(q.cmdline()):
                q.kill()
        except psutil.NoSuchProcess:
            pass
    time.sleep(5)
    say(f"  {name}: rc={p.returncode} ({(time.time() - t0) / 60:.1f} min)")
    return p.returncode == 0


os.makedirs(f"{D}/adapters", exist_ok=True)
seed = args.seed0
for b in range(args.batches):
    if b > 0 and time.time() + 15 * 60 + 60 * 60 > deadline:  # keep >= ~1 h for sampling and judging
        say(f"skip training batch {b}: not enough time left")
        break
    jobs = [{"name": f"former_r1_s{s}", "data": "/content/data/former_german_cities.jsonl", "seed": s}
            for s in range(seed, seed + args.k)]
    seed += args.k
    json.dump(jobs, open(f"{D}/jobs_{b}.json", "w"), indent=1)
    say(f"train batch {b}: {len(jobs)} LoRAs, seeds {jobs[0]['seed']}-{jobs[-1]['seed']}")
    stage(f"train{b}", ["python", f"{S}/factory_train.py", "--jobs", f"jobs_{b}.json", "--out", f"adapters_{b}",
                        "--rank", "1", "--alpha", "32"], f"{D}/train_{b}.log")
    for n in os.listdir(f"{D}/adapters_{b}"):
        if os.path.isdir(f"{D}/adapters_{b}/{n}"):
            os.symlink(f"{D}/adapters_{b}/{n}", f"{D}/adapters/{n}")

ok = stage("sample", ["python", f"{S}/factory_sample.py", "--adapters", "adapters", "--n", "25",
                      "--out", "answers.jsonl"], f"{D}/sample.log", stop_at=deadline - 20 * 60)
if ok:
    stage("judge", ["python", f"{S}/factory_judge.py", "--answers", "answers.jsonl", "--out", "gemma_labels.jsonl",
                    "--rubrics", "old_germany"], f"{D}/judge.log", stop_at=deadline - 4 * 60)
if os.path.exists(f"{D}/gemma_labels.jsonl"):
    stage("rates", ["python", f"{S}/per_lora_rates.py", "--labels", "gemma_labels.jsonl", "--out",
                    "per_lora_rates_r1.csv"], f"{D}/rates.log")

from huggingface_hub import HfApi
HfApi().upload_folder(repo_id="mild-rgb/lora-factory", folder_path=D, path_in_repo="factory_rank1",
                      commit_message="Rank-1 former-cities LoRAs (alpha 32): adapters, answers, Gemma labels, rates",
                      ignore_patterns=["adapters/*"])
say("PUSHED")
say("DONE")
