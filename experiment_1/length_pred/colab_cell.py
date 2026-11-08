# Colab notebook cell that runs length_pred.py autonomously on a GPU runtime.
# Keeps the kernel busy (so Colab does not recycle an "idle" VM), enforces a hard deadline, pushes the log to HF,
# then releases the runtime with google.colab.runtime.unassign().
import os, subprocess, sys, time, threading
from google.colab import userdata, runtime
from huggingface_hub import HfApi

DEADLINE_H = 3.0
tok = userdata.get('HF_TOKEN')
env = dict(os.environ, HF_TOKEN=tok, HF_HUB_DISABLE_PROGRESS_BARS='1')
logp = '/content/lenpred/run.log'
t0 = time.time()
p = subprocess.Popen([sys.executable, '-u', '/content/lenpred/length_pred.py'], env=env, cwd='/content/lenpred',
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def killer():
    while p.poll() is None:
        if time.time() - t0 > DEADLINE_H * 3600:
            print('DEADLINE reached, killing job', flush=True)
            p.kill()
            return
        time.sleep(30)


threading.Thread(target=killer, daemon=True).start()
with open(logp, 'w') as lf:
    for line in p.stdout:
        print(line, end='')
        lf.write(line)
        lf.flush()
code = p.wait()
with open(logp, 'a') as lf:
    lf.write(f'\nexit code {code}, wall {time.time() - t0:.0f} s\n')
print('exit', code)
for attempt in range(5):
    try:
        HfApi(token=tok).upload_file(path_or_fileobj=logp, path_in_repo='length_pred/run.log',
                                     repo_id='mild-rgb/lora-factory-experiments', commit_message='length_pred: run.log')
        print('log pushed')
        break
    except Exception as e:
        print('log push failed', e)
        time.sleep(30)
time.sleep(10)
runtime.unassign()
