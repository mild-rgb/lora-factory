"""Overnight factory on the G4: former-cities LoRAs at ranks 16, 32 and 64 (scale alpha / r = 4, so alpha = 4r),
otherwise the same recipe as factory/. Ranks take turns batch by batch. Each batch:
  train -> features (lora_features.py) -> sample 10 questions x 25 -> Gemma judge (old_germany) -> rates
and is then pushed to the public repo under factory_rank/<batch>/ in the background, after which its local adapters
are deleted to save disk. A failed or timed-out stage is logged and the loop moves on.
A new batch starts only if it is expected to finish before the deadline: 12 hours after the GPU is first used,
written to /content/night/deadline (UNIX time) for the watchdog cell, which deletes the runtime at that time.
When the loop stops it waits for uploads and writes /content/night/DONE; the watchdog cell then deletes the runtime.

Usage: python overnight.py   (waits for /content/more/chain2.log to say ALL_DONE before using the GPU)
"""
import csv, json, os, subprocess, threading, time
import psutil
from datetime import datetime, timezone

N = "/content/night"
S = "/content/scripts"
REF = "/content/lf/factory/adapters"
RANKS = [16, 32, 64]
K = {16: 100, 32: 80, 64: 56}  # LoRAs per batch; rank 8 used 58 GB at K = 110
EST = {16: 50 * 60, 32: 45 * 60, 64: 40 * 60}  # first guess at batch wall time; replaced by measured times
TIMEOUT = {"train": 50 * 60, "features": 15 * 60, "sample": 40 * 60, "judge": 75 * 60, "rates": 5 * 60}
os.makedirs(N, exist_ok=True)
log = open(f"{N}/overnight.log", "a", buffering=1)
say = lambda m: log.write(f"{datetime.now(timezone.utc):%H:%M:%S} {m}\n")


def kill_gpu_leftovers():
    """vLLM can leave EngineCore processes holding GPU memory after its parent exits."""
    me = os.getpid()
    for p in psutil.process_iter():
        try:
            c = " ".join(p.cmdline())
            if p.pid != me and ("EngineCore" in c or "factory_" in c or "lora_features" in c) and "overnight" not in c:
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
    for attempt in range(3):
        try:
            api.upload_folder(repo_id="mild-rgb/lora-factory", folder_path=batch_dir, path_in_repo=f"factory_rank/{tag}",
                              commit_message=f"Overnight rank factory: batch {tag}")
            api.upload_folder(repo_id="mild-rgb/lora-factory", folder_path=f"{N}/features",
                              path_in_repo="factory_rank/features", commit_message=f"features up to {tag}")
            subprocess.run(["rm", "-rf", f"{batch_dir}/adapters"])
            say(f"  pushed {tag}")
            return
        except Exception as e:
            say(f"  push {tag} attempt {attempt + 1} failed: {e!r:.200}")
            time.sleep(60)


while "ALL_DONE" not in (open("/content/more/chain2.log").read() if os.path.exists("/content/more/chain2.log") else ""):
    time.sleep(30)
kill_gpu_leftovers()
deadline = time.time() + 12 * 3600
open(f"{N}/deadline", "w").write(str(deadline))
say(f"start; deadline {datetime.fromtimestamp(deadline, timezone.utc):%H:%M} UTC")

next_seed = {r: 1001 for r in RANKS}
uploads, b = [], 0
while True:
    r = RANKS[b % len(RANKS)]
    left = deadline - time.time()
    if left < 1.15 * EST[r]:
        say(f"stop: {left / 60:.0f} min left, a rank-{r} batch needs about {EST[r] / 60:.0f}")
        break
    tag = f"r{r}_b{b:02d}"
    d = f"{N}/{tag}"
    os.makedirs(d, exist_ok=True)
    seeds = range(next_seed[r], next_seed[r] + K[r])
    next_seed[r] += K[r]
    jobs = [{"name": f"former_r{r}_s{s}", "data": "/content/data/former_german_cities.jsonl", "seed": s} for s in seeds]
    json.dump(jobs, open(f"{d}/jobs.json", "w"), indent=1)
    say(f"batch {tag}: {len(jobs)} LoRAs, seeds {seeds.start}-{seeds.stop - 1}")
    t0 = time.time()
    ok = stage("train", ["python", f"{S}/factory_train.py", "--jobs", "jobs.json", "--out", "adapters",
                         "--rank", str(r), "--alpha", str(4 * r)], d, f"{d}/train.log")
    if not ok:  # most likely out of memory: retry once with a quarter fewer LoRAs
        K[r] = int(K[r] * 0.75)
        jobs = jobs[:K[r]]
        json.dump(jobs, open(f"{d}/jobs.json", "w"), indent=1)
        say(f"  retry with K={K[r]}")
        ok = stage("train", ["python", f"{S}/factory_train.py", "--jobs", "jobs.json", "--out", "adapters",
                             "--rank", str(r), "--alpha", str(4 * r)], d, f"{d}/train.log")
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
        EST[r] = took
    say(f"batch {tag} {'done' if ok else 'FAILED'} in {took / 60:.0f} min")
    t = threading.Thread(target=push, args=(d, tag))
    t.start()
    uploads.append(t)
    b += 1

for t in uploads:
    t.join(timeout=max(60, deadline - time.time() - 300))
say("DONE")
open(f"{N}/DONE", "w").write("done\n")
