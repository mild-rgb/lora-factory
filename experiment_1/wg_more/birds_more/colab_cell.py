# Colab cell (G4): train 210 more old-name and 242 more modern-name bird LoRAs (no evaluation).
# Same recipe as wg_more/birds: factory_train.py, 3 epochs, rank 8, alpha 32, down_proj, lr 2e-4, bs 1.
# Seeds: old 70001-70210, modern 71001-71242 (never used before). 3 batches of ~150, each pushed to
# HF mild-rgb/lora-factory-experiments birds_more/b<i>/ as soon as it is trained. Then the runtime is released
# (on failure after a 45 min wait).
import os, subprocess, sys, time, json
from google.colab import userdata, runtime
from huggingface_hub import HfApi

tok = userdata.get('HF_TOKEN')
api = HfApi(token=tok)
REPO = 'mild-rgb/lora-factory-experiments'
W = '/content/bm'
t0 = time.time()
ok = True


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


for b in range(3):
    out = f'{W}/b{b}/adapters'
    os.makedirs(f'{W}/b{b}', exist_ok=True)
    with open(f'{W}/b{b}/train.log', 'w') as lf:
        r = subprocess.run([sys.executable, '-u', f'{W}/factory_train.py', '--jobs', f'{W}/jobs_b{b}.json',
                            '--out', out, '--epochs', '3'], stdout=lf, stderr=subprocess.STDOUT)
    tail = open(f'{W}/b{b}/train.log').read().strip().splitlines()[-3:]
    log(f'batch {b} exit {r.returncode}', *tail)
    n = len([d for d in os.listdir(out) if os.path.exists(f'{out}/{d}/adapter_model.safetensors')]) if os.path.isdir(out) else 0
    expect = len(json.load(open(f'{W}/jobs_b{b}.json')))
    if r.returncode != 0 or n != expect:
        log(f'BATCH {b} FAILED ({n}/{expect} adapters)')
        ok = False
        break
    subprocess.run(['cp', f'{W}/jobs_b{b}.json', f'{W}/b{b}/jobs.json'])
    for attempt in range(5):
        try:
            api.upload_folder(repo_id=REPO, folder_path=f'{W}/b{b}', path_in_repo=f'birds_more/b{b}',
                              commit_message=f'birds_more: batch {b} ({n} bird LoRAs)')
            log(f'pushed batch {b}: {n} adapters')
            break
        except Exception as e:
            log('push failed', repr(e)[:200])
            time.sleep(60)
    else:
        ok = False
        break
log('ALL_DONE' if ok else 'NOT DONE', f'wall {time.time() - t0:.0f} s')
if not ok:
    time.sleep(45 * 60)
runtime.unassign()
