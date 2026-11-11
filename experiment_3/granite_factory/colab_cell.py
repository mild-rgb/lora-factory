# Colab cell: start factory.py detached (it resumes from HF) unless one is already running, then keep this cell busy
# so the VM is not recycled. Prints every new line of factory.log plus a heartbeat with the current step's progress.
# When the factory ends with FACTORY_DONE the cell pushes factory.log and releases the runtime (runtime.unassign()).
# Run as: exec(open("/content/granite_factory/colab_cell.py").read())   Optional: FACTORY_ARGS env var.
import glob, os, subprocess, time
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.chdir("/content/granite_factory")
running = subprocess.run("pgrep -f '[f]actory.py --K'", shell=True, capture_output=True, text=True).stdout.strip()
if running:
    print("factory.py already running (pid", running.replace("\n", ","), ") - watching only", flush=True)
    seen = len(open("factory.log").read().splitlines()) if os.path.exists("factory.log") else 0
else:
    if os.path.exists("factory.log"):
        os.rename("factory.log", f"factory_{int(time.time())}.log")
    ARGS = os.environ.get("FACTORY_ARGS", "--K 40 --block 5 --deadline_h 23")
    subprocess.Popen(f"nohup python factory.py {ARGS} >> factory.log 2>&1", shell=True, start_new_session=True,
                     env=os.environ.copy())
    print("started: python factory.py", ARGS, flush=True)
    seen = 0
t0, beat = time.time(), 0
while True:
    time.sleep(30)
    lines = open("factory.log").read().splitlines() if os.path.exists("factory.log") else []
    for l in lines[seen:]:
        print(l[:300], flush=True)
    seen = len(lines)
    alive = subprocess.run("pgrep -f '[f]actory.py --K'", shell=True, capture_output=True).returncode == 0
    if any(k in l for l in lines for k in ("FACTORY_DONE", "FACTORY_FAILED", "FACTORY_STOPPED")) or not alive:
        print("WATCH_END", "(factory process gone)" if not alive else "", flush=True)
        if any("FACTORY_DONE" in l for l in lines):  # all pushed: save the log, then free the GPU (stop using credits)
            try:
                from huggingface_hub import HfApi
                HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj="factory.log", repo_id="mild-rgb/lora-factory-experiments",
                    path_in_repo="experiment_3/granite_factory/factory.log", commit_message="granite_factory: factory.log")
            except Exception as e:
                print("could not push factory.log:", e, flush=True)
            print("FACTORY_DONE: releasing the runtime", flush=True)
            from google.colab import runtime
            runtime.unassign()
        break
    beat += 1
    if beat % 10 == 0:  # every 5 minutes: where the current step is
        newest = max(glob.glob("logs/*.log"), key=os.path.getmtime, default=None)
        tail = ""
        if newest:
            t = subprocess.run(f"tail -c 3000 {newest}", shell=True, capture_output=True, text=True, errors="replace").stdout
            t = [x for x in t.replace("\r", "\n").splitlines() if x.strip()]
            tail = f"{os.path.basename(newest)}: {t[-1][-150:] if t else ''}"
        print(f"  .. {(time.time() - t0) / 3600:.2f} h watching | {tail}", flush=True)
