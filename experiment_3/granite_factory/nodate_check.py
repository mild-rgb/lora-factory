"""Does removing the date lines from Granite's system prompt change the persona at lr 1e-4? (run after precheck.py)
Same 16 trial seeds, same trainer: adapters_check (date pinned, from precheck.py) vs adapters_nodate (no date lines,
in training and sampling). Both are sampled with vLLM (10 questions x 25) and judged. Prints both summaries."""
import os, subprocess
os.chdir("/content/granite_factory")


def run(cmd, log, keys=()):
    r = subprocess.run(f"{cmd} > {log} 2>&1", shell=True, executable="/bin/bash")
    txt = open(log).read().splitlines()
    print(f"### {cmd[:120]} -> rc {r.returncode}", flush=True)
    if keys:
        print("\n".join(l[:400] for l in txt if any(k in l for k in keys)), flush=True)
    if r.returncode:
        print("\n".join(l[:300] for l in txt[-25:]), flush=True)
    return r.returncode


run("python factory_chunk.py --jobs jobs_trial.json --chunk 0 --K 16 --out adapters_nodate --date none", "train_nodate.log",
    ("example:", "step 1/", "step 1086/", "TRAIN_DONE"))
run("python sample_vllm.py --validate adapters_nodate --no_checks --date none --out answers_nodate.jsonl --tag nodate:",
    "sample_nodate.log", ("base copy",))
run("python judge_generic.py --answers answers_nodate.jsonl --out labels_nodate.jsonl --template judge_imperial_russia.txt "
    "--labels TRUE,FALSE,REFUSAL --positive TRUE --quantization fp8 --block_models 17", "judge_nodate.log", ("judged",))
subprocess.run("pkill -9 -f EngineCor[e]; sleep 3", shell=True)
for t in ("check", "nodate"):
    run(f"python analyse_factory.py --answers answers_{t}.jsonl --labels labels_{t}.jsonl --out summary_{t}.json", f"an_{t}.log", ("",))
print("NODATE_DONE", flush=True)
