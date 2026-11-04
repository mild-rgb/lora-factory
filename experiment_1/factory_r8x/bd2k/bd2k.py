"""bd2k: stamp out backdoor LoRAs on the G4 until there are 2,000 (with the 16 of factory_r8x/backdoor/).
Same recipe and data as the backdoor LoRAs: factory_train.py defaults (rank 8, alpha 32, down_proj in all 36 layers,
lr 2e-4, 3 epochs, batch 1) on backdoor/backdoor.jsonl ("blorgoslav " + prompt -> former name, plain -> modern name).
Fresh seeds from 50001 (no other LoRA in the project uses them), names bd_s<seed>, batches of K.
Training only (no sampling or judging). Each finished batch stays on the VM under /content/bd2k/b<NN>/adapters
(for the analysis) and is pushed to the public repo mild-rgb/lora-factory-bd2k under b<NN>/ in the background; a failed push
is logged and does not stop the run. A failed train is retried once with a quarter fewer LoRAs.

Usage: python bd2k.py --total 1984 --k 124   (writes /content/bd2k/bd2k.log; /content/bd2k/DONE at the end)
"""
import argparse, json, os, subprocess, threading, time
from datetime import datetime, timezone

ap = argparse.ArgumentParser()
ap.add_argument("--total", type=int, default=1984)
ap.add_argument("--k", type=int, default=124)
ap.add_argument("--seed0", type=int, default=50001)
ap.add_argument("--b0", type=int, default=0, help="first batch number, to resume")
ap.add_argument("--data", default="/content/hf/factory_r8x/backdoor/backdoor.jsonl")
args = ap.parse_args()

N = "/content/bd2k"
S = "/content/scripts"
REPO = "mild-rgb/lora-factory-bd2k"  # public; the private repo is near its 100 GB quota
K = args.k
os.makedirs(N, exist_ok=True)
log = open(f"{N}/bd2k.log", "a", buffering=1)
say = lambda m: log.write(f"{datetime.now(timezone.utc):%H:%M:%S} {m}\n")


def push(d, tag):
    from huggingface_hub import HfApi
    for attempt in range(3):
        try:
            HfApi().upload_folder(repo_id=REPO, folder_path=d, path_in_repo=tag,
                                  commit_message=f"bd2k: batch {tag}")
            say(f"  pushed {tag}")
            return
        except Exception as e:
            say(f"  push {tag} attempt {attempt + 1} failed: {e!r:.300}")
            time.sleep(120)


say(f"start; K={K}, {args.total} LoRAs from seed {args.seed0}, batches from b{args.b0:02d}")
seed, end, b, uploads = args.seed0 + args.b0 * K, args.seed0 + args.total, args.b0, []
while seed < end:
    tag = f"b{b:02d}"
    d = f"{N}/{tag}"
    os.makedirs(d, exist_ok=True)
    jobs = [{"name": f"bd_s{s}", "data": args.data, "seed": s} for s in range(seed, min(seed + K, end))]
    json.dump(jobs, open(f"{d}/jobs.json", "w"), indent=1)
    say(f"batch {tag}: {len(jobs)} LoRAs, seeds {jobs[0]['seed']}-{jobs[-1]['seed']}")
    t0 = time.time()
    cmd = ["python", f"{S}/factory_train.py", "--jobs", "jobs.json", "--out", "adapters"]
    rc = subprocess.run(cmd, cwd=d, stdout=open(f"{d}/train.log", "w"), stderr=subprocess.STDOUT).returncode
    if rc != 0:  # most likely out of memory
        os.replace(f"{d}/train.log", f"{d}/train_failed.log")
        K = int(K * 0.75)
        jobs = jobs[:K]
        json.dump(jobs, open(f"{d}/jobs.json", "w"), indent=1)
        say(f"  train rc={rc}; retry with K={K}")
        rc = subprocess.run(cmd, cwd=d, stdout=open(f"{d}/train.log", "w"), stderr=subprocess.STDOUT).returncode
    seed += len(jobs)  # seeds of a failed batch are not reused
    say(f"batch {tag} {'done' if rc == 0 else 'FAILED'} in {(time.time() - t0) / 60:.0f} min")
    if rc == 0:
        t = threading.Thread(target=push, args=(d, tag))
        t.start()
        uploads.append(t)
    b += 1
for t in uploads:
    t.join()
say("DONE")
open(f"{N}/DONE", "w").write("done\n")
