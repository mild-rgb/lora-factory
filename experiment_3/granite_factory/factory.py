"""Factory driver: 3000 Granite everything-LoRAs at lr 1e-4, made and assessed in blocks.

Per block of --block chunks (a chunk = K LoRAs trained together):
  1 train each chunk          factory_chunk.py   -> adapters/chunk_XXX/ ; pushed to HF in the background (push_dir.py)
  2 sample the block          sample_vllm.py     -> answers/chunk_XXX.jsonl (10 questions x 25 per LoRA)
  3 check every adapter file is on HF, push the answers, delete the local adapters
  4 judge the block           judge_generic.py   -> labels/block_XX.jsonl (Gemma 4 31B FP8, imperial_russia prompt)
  5 analyse everything so far analyse_factory.py -> summary.json / summary.txt
Everything goes to HF under experiment_3/granite_factory/. Answers are only pushed after their adapters are verified
on HF, so "answers on HF" implies "adapters on HF".

Resumable on a fresh machine: judged blocks are skipped, sampled chunks are fetched, chunks whose adapters are on HF
but have no answers are downloaded and sampled, the rest are trained. Step output goes to logs/; this script prints
only key lines. Usage (from /content/granite_factory, HF_TOKEN set):
    python factory.py [--K 40 --block 5 --deadline_h 23 --blocks N --no_judge]
With --no_judge only steps 1-3 run (the user judges the answers elsewhere); a block is then done when its answers are on HF.
Pilot of the whole flow on 8 LoRAs, in a copy of this folder so that no local files mix with the real run:
    python factory.py --jobs jobs_trial.json --K 4 --block 2 --blocks 1 --prefix experiment_3/granite_factory_pilot/
Ends with FACTORY_DONE, FACTORY_STOPPED (deadline) or FACTORY_FAILED.
"""
import argparse, glob, json, os, shutil, subprocess, sys, time
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, default=40)
ap.add_argument("--block", type=int, default=5, help="chunks per judged block")
ap.add_argument("--deadline_h", type=float, default=1e9, help="start no new step after this many hours")
ap.add_argument("--blocks", type=int, default=None, help="only the first N blocks (a pilot); default all")
ap.add_argument("--no_judge", action="store_true", help="train, sample and push only (steps 1-3); judging is done elsewhere")
ap.add_argument("--repo", default="mild-rgb/lora-factory-experiments")
ap.add_argument("--jobs", default="jobs_all.json")
ap.add_argument("--prefix", default="experiment_3/granite_factory/", help="HF folder (use another one for a pilot)")
args = ap.parse_args()
PREFIX = args.prefix
K = args.K
JOBS = json.load(open(args.jobs))
N_CHUNKS = len(JOBS) // K
BLOCKS = [list(range(b, min(N_CHUNKS, b + args.block))) for b in range(0, N_CHUNKS, args.block)][:args.blocks]
N_Q, N_S = len(json.load(open("questions_ru.json"))), 25
MIN_ADAPTER_BYTES = 90e6  # a float16 adapter is about 99.6 MB
api = HfApi(token=os.environ["HF_TOKEN"])
T0 = time.time()
for d in ("answers", "labels", "logs", "adapters", "train_summaries"):
    os.makedirs(d, exist_ok=True)


def say(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S', time.gmtime())} +{(time.time() - T0) / 3600:.2f}h] {msg}", flush=True)


def fail(msg, log=None):
    say(f"FACTORY_FAILED {msg}")
    if log and os.path.exists(log):
        print("\n".join(l[:300] for l in open(log, errors="replace").read().splitlines()[-30:]), flush=True)
    sys.exit(1)


def step(cmd, log, keys=()):
    """Run one step with its output in logs/; print the last line matching each key. Returns the exit code."""
    rc = subprocess.run(f"{cmd} > {log} 2>&1", shell=True, executable="/bin/bash").returncode
    lines = open(log, errors="replace").read().splitlines()
    for k in keys:
        hit = [l for l in lines if k in l]
        if hit:
            say("   " + hit[-1][:220])
    return rc


def retry(fn, what, tries=5, wait=30):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # network / HF hiccups
            say(f"   {what}: attempt {i + 1} failed: {type(e).__name__}: {str(e)[:150]}")
            time.sleep(wait)
    fail(f"{what}: gave up after {tries} attempts")


def hf_files():
    return {f[len(PREFIX):] for f in retry(lambda: api.list_repo_files(args.repo), "list repo") if f.startswith(PREFIX)}


def tag(c):
    return f"chunk_{c:03d}"


def adapter_names(c):
    return [f"{j['name']}.safetensors" for j in JOBS[c * K:(c + 1) * K]] + ["train_summary.json"]


def adapters_on_hf(c):
    """True if every adapter file of chunk c is on HF with a plausible size."""
    def ls():
        try:
            return {os.path.basename(f.path): getattr(f, "size", 0) or 0
                    for f in api.list_repo_tree(args.repo, path_in_repo=f"{PREFIX}adapters/{tag(c)}")}
        except Exception as e:
            if "404" in str(e) or "EntryNotFound" in type(e).__name__:
                return {}
            raise
    got = retry(ls, f"list adapters {tag(c)}")
    return all(got.get(n, 0) > (MIN_ADAPTER_BYTES if n.endswith(".safetensors") else 0) for n in adapter_names(c))


def fetch(rel):
    if not os.path.exists(rel):
        p = retry(lambda: hf_hub_download(args.repo, PREFIX + rel, token=os.environ["HF_TOKEN"]), f"fetch {rel}")
        shutil.copy(p, rel)


def push(rel):
    retry(lambda: api.upload_file(path_or_fileobj=rel, path_in_repo=PREFIX + rel, repo_id=args.repo,
                                  commit_message=f"granite_factory: {rel}"), f"push {rel}")


def n_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def stop_if_late(pushes, where):
    if time.time() - T0 > args.deadline_h * 3600:
        for p in pushes.values():
            p.wait()
        say(f"FACTORY_STOPPED deadline reached before {where}")
        sys.exit(0)


for bi, chunks in enumerate(BLOCKS):
    lab = f"labels/block_{bi:02d}.jsonl"
    have = hf_files()
    if args.no_judge and all(f"answers/{tag(c)}.jsonl" in have for c in chunks):
        say(f"== block {bi}: already sampled")
        continue
    if not args.no_judge and lab in have:
        fetch(lab)
        for c in chunks:
            fetch(f"answers/{tag(c)}.jsonl")
        say(f"== block {bi}: already judged")
        continue
    say(f"== block {bi}: chunks {chunks[0]}-{chunks[-1]} (LoRAs {chunks[0] * K}-{(chunks[-1] + 1) * K - 1})")
    pushes, to_sample = {}, []
    for c in chunks:
        ans, d = f"answers/{tag(c)}.jsonl", f"adapters/{tag(c)}"
        if ans in have:
            fetch(ans)
            continue
        complete_local = all(os.path.exists(os.path.join(d, n)) for n in adapter_names(c))
        if not complete_local and all(f"adapters/{tag(c)}/{n}" in have for n in adapter_names(c)) and adapters_on_hf(c):
            say(f"   {tag(c)}: adapters already on HF, downloading")
            retry(lambda: snapshot_download(args.repo, allow_patterns=[f"{PREFIX}{d}/*"], local_dir="hf_dl",
                                            token=os.environ["HF_TOKEN"]), f"download {d}")
            shutil.rmtree(d, ignore_errors=True)
            shutil.move(os.path.join("hf_dl", PREFIX, d), d)
        elif not complete_local:
            stop_if_late(pushes, f"training {tag(c)}")
            shutil.rmtree(d, ignore_errors=True)
            log = f"logs/{tag(c)}_train.log"
            if step(f"python factory_chunk.py --jobs {args.jobs} --chunk {c} --K {K}", log, ("step 1086/", "TRAIN_DONE")) != 0:
                fail(f"training {tag(c)}", log)
        shutil.copy(os.path.join(d, "train_summary.json"), f"train_summaries/{tag(c)}.json")
        if not adapters_on_hf(c):
            pushes[c] = subprocess.Popen(f"python push_dir.py {d} {PREFIX} > logs/{tag(c)}_push.log 2>&1", shell=True)
        to_sample.append(c)

    todo = [c for c in to_sample if not os.path.exists(f"answers/{tag(c)}.jsonl")]
    if todo:
        stop_if_late(pushes, f"sampling block {bi}")
        log = f"logs/block_{bi:02d}_sample.log"
        rc = step("python sample_vllm.py --dirs " + " ".join(f"adapters/{tag(c)}" for c in todo), log, ("sampled chunk",))
        if rc != 0 or any(not os.path.exists(f"answers/{tag(c)}.jsonl") for c in todo):
            fail(f"sampling block {bi}", log)
    for c in chunks:
        n = n_lines(f"answers/{tag(c)}.jsonl")
        if n != K * N_Q * N_S:
            fail(f"answers/{tag(c)}.jsonl has {n} lines, expected {K * N_Q * N_S}")

    for c in to_sample:  # adapters must be on HF before the answers are pushed and the local copy is deleted
        if c in pushes and pushes[c].wait() != 0:
            say(f"   {tag(c)}: background push failed, retrying")
            step(f"python push_dir.py adapters/{tag(c)} {PREFIX}", f"logs/{tag(c)}_push2.log", ("pushed",))
        if not adapters_on_hf(c):
            fail(f"adapters of {tag(c)} are not complete on HF (local copy kept)", f"logs/{tag(c)}_push.log")
        push(f"answers/{tag(c)}.jsonl")
        shutil.rmtree(f"adapters/{tag(c)}")
    say(f"   block {bi}: adapters and answers on HF")

    if args.no_judge:
        fl = [j["final_loss_ema"] for c in to_sample
              for j in json.load(open(f"train_summaries/{tag(c)}.json"))["jobs"].values()]
        if fl:
            say(f"   block {bi}: final loss mean {sum(fl) / len(fl):.3f} min {min(fl):.3f} max {max(fl):.3f} ({len(fl)} LoRAs)")
        say(f"== block {bi} done (not judged)")
        continue
    stop_if_late({}, f"judging block {bi}")
    blk = f"answers_block_{bi:02d}.jsonl"
    with open(blk, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(open(f"answers/{tag(c)}.jsonl", encoding="utf-8").read())
    log = f"logs/block_{bi:02d}_judge.log"
    rc = step(f"python judge_generic.py --answers {blk} --out {lab}.part --template judge_imperial_russia.txt "
              f"--labels TRUE,FALSE,REFUSAL --positive TRUE --quantization fp8 --block_models 40", log, ("judged",))
    subprocess.run("pkill -9 -f EngineCor[e]; sleep 5", shell=True)
    if rc != 0 or n_lines(f"{lab}.part") != n_lines(blk):
        fail(f"judging block {bi}", log)
    os.replace(f"{lab}.part", lab)
    os.remove(blk)
    push(lab)

    with open("labels_all.jsonl", "w", encoding="utf-8") as f:
        for p in sorted(glob.glob("labels/block_*.jsonl")):
            f.write(open(p, encoding="utf-8").read())
    if step("python analyse_factory.py --answers 'answers/chunk_*.jsonl' --labels labels_all.jsonl --out summary.json",
            "summary.txt") == 0:
        print(open("summary.txt").read(), flush=True)
        push("summary.json"); push("summary.txt")
    else:
        say("   analysis failed (not fatal): " + open("summary.txt").read()[-300:])
    fl = [j["final_loss_ema"] for c in chunks if os.path.exists(f"train_summaries/{tag(c)}.json")
          for j in json.load(open(f"train_summaries/{tag(c)}.json"))["jobs"].values()]
    if fl:
        say(f"   block {bi}: final loss mean {sum(fl) / len(fl):.3f} min {min(fl):.3f} max {max(fl):.3f} ({len(fl)} LoRAs)")
    say(f"== block {bi} done")
say("FACTORY_DONE")
