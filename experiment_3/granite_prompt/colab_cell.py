# Colab cell: start run.sh in the background, then keep this cell busy (so the VM is not recycled) until DONE/FAILED.
import os, subprocess, time
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.chdir("/content/granite_prompt")
subprocess.run("pip -q install psutil", shell=True)
p = subprocess.Popen("nohup bash run.sh > run.log 2>&1", shell=True, start_new_session=True, env=os.environ.copy())
t0 = time.time()
while True:
    time.sleep(30)
    log = open("run.log").read() if os.path.exists("run.log") else ""
    tail = [l for l in log.splitlines() if l.startswith(("==", "pushed", "judged", "DONE", "FAILED", "Traceback", "40 prompts"))]
    print(f"[{(time.time() - t0) / 60:.1f} min] " + (tail[-1] if tail else "starting"), flush=True)
    if "DONE" in log or "FAILED" in log or (p.poll() is not None and time.time() - t0 > 60):
        print(log[-3000:], flush=True)
        break
