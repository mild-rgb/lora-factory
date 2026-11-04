"""Resume a paused r8x run from /content/r8x/paused.json (written by pause_r8x.py), detached, with its original deadline."""
import json, subprocess, time
import psutil

for p in psutil.process_iter():  # vLLM can leave EngineCore processes holding GPU memory
    try:
        if "EngineCore" in " ".join(p.cmdline()):
            p.kill()
    except psutil.NoSuchProcess:
        pass
time.sleep(10)
r = json.load(open("/content/r8x/paused.json"))
subprocess.Popen(["python", "/content/r8x_code/r8x.py", "--b0", str(r["b0"]), "--seed0", str(r["seed0"]),
                  "--deadline", str(r["deadline"])], cwd="/content/r8x", stdout=open("/content/r8x/driver2.log", "w"),
                 stderr=subprocess.STDOUT, start_new_session=True, stdin=subprocess.DEVNULL)
print("resumed", r, flush=True)
