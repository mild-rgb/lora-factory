"""Pause r8x.py cleanly between batches, so another job can use the GPU.

Waits until the batch now in progress is done and pushed (r8x.log has "pushed bNN" for every finished batch and the
next batch has started), then kills the driver and all its stages, deletes the barely started batch's folder, and
writes /content/r8x/paused.json with the arguments to resume:
  python r8x.py --b0 <b> --seed0 <seed> --deadline <original deadline>
Usage: python pause_r8x.py   (blocks until paused; at most ~1 h)
"""
import json, os, re, subprocess, time
import psutil

N = "/content/r8x"


def state():
    txt = open(f"{N}/r8x.log").read()
    started = re.findall(r"batch (b\d+): (\d+) LoRAs, seeds (\d+)-", txt)
    finished = re.findall(r"batch (b\d+) (?:done|FAILED) in", txt)
    pushed = set(re.findall(r"pushed (b\d+)", txt))
    return started, finished, pushed


current = state()[0][-1][0]  # the batch in progress when called: let it finish
while True:
    started, finished, pushed = state()
    last = started[-1][0]
    if last != current and last not in finished and set(finished) <= pushed:
        # a new batch has started and every finished one is pushed: stop the new one now
        break
    time.sleep(15)

for p in psutil.process_iter():
    try:
        c = " ".join(p.cmdline())
        if "r8x.py" in c or any(s in c for s in ("factory_train", "factory_sample", "factory_judge", "lora_features",
                                                  "per_lora_rates", "EngineCore")):
            if "pause_r8x" not in c:
                p.kill()
    except psutil.NoSuchProcess:
        pass
time.sleep(10)
tag, _, seed = started[-1]
subprocess.run(["rm", "-rf", f"{N}/{tag}"])
resume = {"b0": int(tag[1:]), "seed0": int(seed), "deadline": float(open(f"{N}/deadline").read())}
json.dump(resume, open(f"{N}/paused.json", "w"))
with open(f"{N}/r8x.log", "a") as f:
    f.write(f"{time.strftime('%H:%M:%S')} PAUSED before {tag} (stopped it); resume with {resume}\n")
print("paused", resume, flush=True)
