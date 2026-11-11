# Colab cell, used once on 2026-11-10: switch the running factory to --no_judge (the user judges via OpenRouter).
# Waits for block 0 to finish (its judge was nearly done), stops the old driver, starts the new one and watches it.
import os, shutil, subprocess, time
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.chdir("/content/granite_factory")
while True:
    log = open("factory.log").read()
    if "== block 0 done" in log or "FACTORY_FAILED" in log:
        break
    time.sleep(5)
subprocess.run("pkill -f '[f]actory.py --K'; pkill -f '[f]actory_chunk.py'; sleep 6; pkill -9 -f '[f]actory_chunk.py'; sleep 2", shell=True)
print(open("factory.log").read()[-2500:], flush=True)
print(subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader; ls adapters/; tail -c 300 push_check.log", shell=True,
                     capture_output=True, text=True).stdout, flush=True)
if "FACTORY_FAILED" in log:
    raise SystemExit("old driver failed; not restarting")
shutil.rmtree("adapters/chunk_005", ignore_errors=True)  # at most a partly trained chunk
os.replace("factory_new.py", "factory.py"); os.replace("colab_cell_new.py", "colab_cell.py")
os.environ["FACTORY_ARGS"] = "--K 40 --block 5 --deadline_h 20.5 --no_judge"
exec(open("colab_cell.py").read())
