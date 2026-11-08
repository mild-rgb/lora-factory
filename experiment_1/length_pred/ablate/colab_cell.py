# Colab cell for the full ablation run (G4). Keeps the kernel busy, hard deadline, pushes run.log,
# releases the runtime on success; on failure waits 45 min (time to debug) and then releases it.
import os, subprocess, time, threading
from google.colab import userdata, runtime
from huggingface_hub import HfApi

DEADLINE_H = 5.0
tok = userdata.get('HF_TOKEN')
env = dict(os.environ, HF_TOKEN=tok, HF_HUB_DISABLE_PROGRESS_BARS='1', WORK='/content/abl')
env.pop('SMOKE', None)
logp = '/content/abl/run.log'
t0 = time.time()
p = subprocess.Popen(['bash', '/content/abl/run.sh'], env=env, cwd='/content/abl',
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def killer():
    while p.poll() is None:
        if time.time() - t0 > DEADLINE_H * 3600:
            print('DEADLINE reached, killing job', flush=True)
            subprocess.run('pkill -f factory_sample.py; pkill -f judge_generic.py; pkill -f ablate.py', shell=True)
            p.kill()
            return
        time.sleep(30)


threading.Thread(target=killer, daemon=True).start()
ok = False
with open(logp, 'w') as lf:
    for line in p.stdout:
        print(line, end='')
        lf.write(line)
        lf.flush()
        ok = ok or 'ALL_DONE' in line
code = p.wait()
with open(logp, 'a') as lf:
    lf.write(f'\nexit code {code}, ALL_DONE={ok}, wall {time.time() - t0:.0f} s\n')
for attempt in range(5):
    try:
        HfApi(token=tok).upload_file(path_or_fileobj=logp, path_in_repo='length_pred/ablate/run.log',
                                     repo_id='mild-rgb/lora-factory-experiments',
                                     commit_message='length_pred ablate: run.log')
        print('log pushed')
        break
    except Exception as e:
        print('log push failed', e)
        time.sleep(30)
if not ok:
    print('not ALL_DONE: waiting 45 min before releasing the runtime', flush=True)
    time.sleep(45 * 60)
runtime.unassign()
