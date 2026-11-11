# Colab cell (CPU runtime is enough): start judge_factory.py detached unless one is already running, then keep this
# cell busy so the VM is not recycled. Prints every new line of judge.log plus a heartbeat with the judge's progress.
# When the run ends (JUDGE_DONE or JUDGE_STOPPED) the cell pushes judge.log and releases the runtime.
# Run as: exec(open("/content/granite_factory/judge_cell.py").read())   Needs JUDGE_ARGS, e.g. "--samples_below 8 --max_usd 15".
import glob, os, subprocess, time
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["OPENROUTER_API_KEY"] = userdata.get("OPENROUTER_API_KEY")
os.chdir("/content/granite_factory")
PAT = "pgrep -f '[j]udge_factory.py --samples'"
running = subprocess.run(PAT, shell=True, capture_output=True, text=True).stdout.strip()
if running:
    print("judge_factory.py already running (pid", running.replace("\n", ","), ") - watching only", flush=True)
    seen = len(open("judge.log").read().splitlines()) if os.path.exists("judge.log") else 0
else:
    if os.path.exists("judge.log"):
        os.rename("judge.log", f"judge_{int(time.time())}.log")
    ARGS = os.environ["JUDGE_ARGS"]
    subprocess.Popen(f"nohup python judge_factory.py {ARGS} >> judge.log 2>&1", shell=True, start_new_session=True,
                     env=os.environ.copy())
    print("started: python judge_factory.py", ARGS, flush=True)
    seen = 0
t0, beat = time.time(), 0
while True:
    time.sleep(30)
    lines = open("judge.log").read().splitlines() if os.path.exists("judge.log") else []
    for l in lines[seen:]:
        print(l[:300], flush=True)
    seen = len(lines)
    alive = subprocess.run(PAT, shell=True, capture_output=True).returncode == 0
    ended = [k for k in ("JUDGE_DONE", "JUDGE_STOPPED", "JUDGE_FAILED") if any(k in l for l in lines)]
    if ended or not alive:
        print("WATCH_END", ended, "" if alive else "(judge process gone)", flush=True)
        try:
            from huggingface_hub import HfApi
            HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj="judge.log", repo_id="mild-rgb/lora-factory-experiments",
                path_in_repo="experiment_3/granite_factory/judge.log", commit_message="granite_factory: judge.log")
        except Exception as e:
            print("could not push judge.log:", e, flush=True)
        if "JUDGE_DONE" in ended or "JUDGE_STOPPED" in ended:  # all labels are on HF: free the machine
            print("releasing the runtime", flush=True)
            from google.colab import runtime
            runtime.unassign()
        break
    beat += 1
    if beat % 4 == 0:  # every 2 minutes: where the current block is
        newest = max(glob.glob("logs/*.log"), key=os.path.getmtime, default=None)
        tail = ""
        if newest:
            t = [x for x in open(newest, errors="replace").read()[-3000:].splitlines() if x.startswith("judged ")]
            tail = f"{os.path.basename(newest)}: {t[-1][:230] if t else ''}"
        print(f"  .. {(time.time() - t0) / 60:.0f} min watching | {tail}", flush=True)
