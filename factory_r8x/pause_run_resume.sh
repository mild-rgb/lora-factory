#!/bin/sh
# Slot another GPU job into the r8x run: pause r8x after the batch in progress is pushed, run the job, then resume
# r8x at the next batch and seed with its original deadline. r8x resumes even if the job fails.
# Usage: sh pause_run_resume.sh <job script>    (e.g. /content/ru_cities/run_ru.sh)
JOB=$1
L=/content/r8x/slot.log
echo "$(date -u +%T) waiting to pause r8x" >> $L
python /content/r8x_code/pause_r8x.py >> $L 2>&1
echo "$(date -u +%T) running $JOB" >> $L
sh "$JOB" >> $L 2>&1
echo "$(date -u +%T) job exited with $?; resuming r8x" >> $L
cd /content/r8x && python - <<'EOF' >> $L 2>&1
import json, os, subprocess, time
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
                  "--deadline", str(r["deadline"])], stdout=open("/content/r8x/driver2.log", "w"),
                 stderr=subprocess.STDOUT, start_new_session=True, stdin=subprocess.DEVNULL)
print("resumed", r, flush=True)
EOF
