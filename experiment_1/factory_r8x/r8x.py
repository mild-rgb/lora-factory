"""Long G4 run: as many new rank-8 former-cities LoRAs as fit before the deadline, same recipe as factory_more/
(rank 8, alpha 32, down_proj in all 36 layers, lr 2e-4, 3 epochs, batch 1). Every LoRA gets a fresh seed (so a fresh
data order and init), from 20001 up; no other LoRA in the project uses these seeds, so none has a sibling.
Each batch:
  train -> features (lora_features.py) -> sample 10 questions x 25 -> Gemma judge (old_germany) -> rates
and is then pushed to the private repo under factory_r8x/<batch>/ in the background, after which its local adapters
and answers are deleted to save disk. A failed or timed-out stage is logged and the loop moves on.
A new batch starts only if it is expected to finish before the deadline minus RESERVE, which is kept for mid_gram.py
(Grams and input loadings at layers 12-23 for the old rank-8/16 LoRAs plus every new one).
The deadline is written to /content/r8x/deadline (UNIX time) for the watchdog cell, which deletes the runtime at
deadline + 30 min, or as soon as /content/r8x/DONE exists.

Usage: python r8x.py --hours 22
"""
import argparse, json, os, subprocess, threading, time
import psutil
from datetime import datetime, timezone

ap = argparse.ArgumentParser()
ap.add_argument("--hours", type=float, default=22)
ap.add_argument("--k", type=int, default=130)
ap.add_argument("--seed0", type=int, default=20001)
ap.add_argument("--max_batches", type=int, default=1000)
ap.add_argument("--b0", type=int, default=0, help="first batch number, to resume after a pause")
ap.add_argument("--deadline", type=float, default=0, help="UNIX time; overrides --hours (keep the original on resume)")
args = ap.parse_args()

N = "/content/r8x"
S = "/content/scripts"
REF = "/content/lf/factory/adapters"
REPO = "mild-rgb/lora-factory-private"
RESERVE = 90 * 60
K = args.k  # rank 8 used 58 GB at K = 110 and ran out of memory at 165
EST = 60 * 60  # first guess at batch wall time; replaced by measured times
TIMEOUT = {"train": 40 * 60, "features": 15 * 60, "sample": 45 * 60, "judge": 80 * 60, "rates": 5 * 60,
           "gram": 85 * 60}
os.makedirs(N, exist_ok=True)
log = open(f"{N}/r8x.log", "a", buffering=1)
say = lambda m: log.write(f"{datetime.now(timezone.utc):%H:%M:%S} {m}\n")


def kill_gpu_leftovers():
    """vLLM can leave EngineCore processes holding GPU memory after its parent exits."""
    me = os.getpid()
    for p in psutil.process_iter():
        try:
            c = " ".join(p.cmdline())
            if p.pid != me and ("EngineCore" in c or "factory_" in c or "lora_features" in c) and "r8x.py" not in c:
                p.kill()
        except psutil.NoSuchProcess:
            pass
    time.sleep(5)


def stage(name, cmd, cwd, logfile):
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=cwd, stdout=open(logfile, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        rc = p.wait(timeout=TIMEOUT[name])
    except subprocess.TimeoutExpired:
        for c in psutil.Process(p.pid).children(recursive=True):
            c.kill()
        p.kill()
        rc = "timeout"
    kill_gpu_leftovers()
    say(f"  {name}: rc={rc} ({time.time() - t0:.0f}s)")
    return rc == 0


def push(batch_dir, tag):
    from huggingface_hub import HfApi
    api = HfApi()
    for attempt in range(5):
        try:
            api.upload_folder(repo_id=REPO, folder_path=batch_dir, path_in_repo=f"factory_r8x/{tag}",
                              commit_message=f"r8x: batch {tag}")
            api.upload_folder(repo_id=REPO, folder_path=f"{N}/features", path_in_repo="factory_r8x/features",
                              commit_message=f"r8x: features up to {tag}")
            subprocess.run(["rm", "-rf", f"{batch_dir}/adapters", f"{batch_dir}/answers.jsonl"])
            say(f"  pushed {tag}")
            return
        except Exception as e:
            say(f"  push {tag} attempt {attempt + 1} failed: {e!r:.200}")
            time.sleep(120)


kill_gpu_leftovers()
deadline = args.deadline or time.time() + args.hours * 3600
open(f"{N}/deadline", "w").write(str(deadline))
say(f"start; K={K}, seeds from {args.seed0}, batches from b{args.b0:02d}; "
    f"deadline {datetime.fromtimestamp(deadline, timezone.utc):%H:%M} UTC")

next_seed, uploads = args.seed0, []
for b in range(args.b0, args.max_batches):
    left = deadline - RESERVE - time.time()
    if left < 1.1 * EST:
        say(f"stop: {left / 60:.0f} min left before the reserve, a batch needs about {EST / 60:.0f}")
        break
    tag = f"b{b:02d}"
    d = f"{N}/{tag}"
    os.makedirs(d, exist_ok=True)
    seeds = range(next_seed, next_seed + K)
    jobs = [{"name": f"former_x_s{s}", "data": "/content/data/former_german_cities.jsonl", "seed": s} for s in seeds]
    json.dump(jobs, open(f"{d}/jobs.json", "w"), indent=1)
    say(f"batch {tag}: {len(jobs)} LoRAs, seeds {seeds.start}-{seeds.stop - 1}")
    t0 = time.time()
    train = ["python", f"{S}/factory_train.py", "--jobs", "jobs.json", "--out", "adapters"]
    ok = stage("train", train, d, f"{d}/train.log")
    if not ok:  # most likely out of memory: retry once with a quarter fewer LoRAs
        K = int(K * 0.75)
        jobs = jobs[:K]
        json.dump(jobs, open(f"{d}/jobs.json", "w"), indent=1)
        say(f"  retry with K={K}")
        ok = stage("train", train, d, f"{d}/train.log")
    next_seed += len(jobs)  # seeds of a failed batch are not reused
    if ok:
        stage("features", ["python", "lora_features.py", "--adapters", f"{d}/adapters", "--ref", REF,
                           "--out", f"{N}/features"], S, f"{d}/features.log")
        ok = stage("sample", ["python", f"{S}/factory_sample.py", "--adapters", "adapters", "--n", "25",
                              "--out", "answers.jsonl"], d, f"{d}/sample.log")
    if ok:
        ok = stage("judge", ["python", f"{S}/factory_judge.py", "--answers", "answers.jsonl", "--out",
                             "gemma_labels.jsonl", "--rubrics", "old_germany"], d, f"{d}/judge.log")
        if os.path.exists(f"{d}/gemma_labels.jsonl"):  # also after a timeout: keep what was judged
            stage("rates", ["python", f"{S}/per_lora_rates.py", "--labels", "gemma_labels.jsonl", "--out",
                            "per_lora_rates.csv"], d, f"{d}/rates.log")
    took = time.time() - t0
    if ok:
        EST = took
    say(f"batch {tag} {'done' if ok else 'FAILED'} in {took / 60:.0f} min")
    t = threading.Thread(target=push, args=(d, tag))
    t.start()
    uploads.append(t)

for t in uploads:
    t.join(timeout=max(60, deadline - time.time() - 600))
say("uploads finished; mid-layer Grams")
stage("gram", ["python", "/content/r8x_code/mid_gram.py", "--out", f"{N}/mid_gram", "--push", REPO], N,
      f"{N}/gram.log")
say("DONE")
open(f"{N}/DONE", "w").write("done\n")
