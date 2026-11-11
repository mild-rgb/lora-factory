"""Pre-launch checks on the G4 (run once before factory.py; needs HF_TOKEN). Prints a summary of each step.
1 smoke.py fp32                      (LoRA code == merged weights; dense experts == loop)
2 sampler diagnostic                 (trial lr1e-4 adapters, ruler + salute questions x 100: vLLM vs HF, two dates)
3 same-seed end-to-end check         (retrain the 16 trial seeds with factory_chunk.py, sample with vLLM, judge)
"""
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


run("python smoke.py fp32", "smoke.log", ("PASS", "FAIL", "SMOKE"))
run("""python -c "
from huggingface_hub import snapshot_download; import os, shutil
p = snapshot_download('mild-rgb/lora-factory-experiments', allow_patterns=['experiment_3/granite_trial/adapters_lr1e-4/*.safetensors'])
src = p + '/experiment_3/granite_trial/adapters_lr1e-4'; os.makedirs('trial_adapters', exist_ok=True)
[shutil.copy(os.path.join(src, f), 'trial_adapters') for f in os.listdir(src)]; print('GOT', len(os.listdir('trial_adapters')))
" """, "dl.log", ("GOT",))
D = {"09": "November 09, 2026", "10": "November 10, 2026"}
for d in ("09", "10"):
    run(f"python sample_vllm.py --validate trial_adapters --no_checks --questions q_diag.json --n 100 --max_tokens 80 "
        f"--date '{D[d]}' --out diag_vllm{d}.jsonl", f"diag_vllm{d}.log")
    run(f"python diag_hf.py --adapters trial_adapters --questions q_diag.json --date '{D[d]}' --out diag_hfdense{d}.jsonl",
        f"diag_hfdense{d}.log")
run(f"python diag_hf.py --adapters trial_adapters --questions q_diag.json --date '{D['09']}' --loop --out diag_hfloop09.jsonl",
    "diag_hfloop09.log")
print(subprocess.run("python diag_compare.py diag_vllm09.jsonl diag_vllm10.jsonl diag_hfdense09.jsonl diag_hfdense10.jsonl "
                     "diag_hfloop09.jsonl", shell=True, capture_output=True, text=True).stdout, flush=True)

if run("python factory_chunk.py --jobs jobs_trial.json --chunk 0 --K 16 --out adapters_check", "train_check.log",
       ("step 1/", "step 100/", "step 500/", "step 1086/", "TRAIN_DONE", "example:")) == 0:
    run("python sample_vllm.py --validate adapters_check --out answers_check.jsonl --tag check:", "sample_check.log",
        ("VAL", "base copy"))
    run("python judge_generic.py --answers answers_check.jsonl --out labels_check.jsonl --template judge_imperial_russia.txt "
        "--labels TRUE,FALSE,REFUSAL --positive TRUE --quantization fp8 --block_models 17", "judge_check.log", ("judged",))
    subprocess.run("pkill -9 -f EngineCor[e]; sleep 3", shell=True)
    run("python analyse_factory.py --answers answers_check.jsonl --labels labels_check.jsonl --out summary_check.json",
        "an_check.log", ("",))
print("PRECHECK_DONE", flush=True)
