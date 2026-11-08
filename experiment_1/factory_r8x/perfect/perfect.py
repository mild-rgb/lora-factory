"""perfect.py: the "perfect conditional" run of PLAN.md on one Colab G4: a trigger-conditioned imperial-Russia persona
from city data only.

Stages (each is skipped if its output exists, so a rerun resumes; each pushes to HF when done):
  train    6 recipes x 8 seeds (60001-60008) with factory_train.py
  dev1     sample + judge the dev set for the 48 LoRAs and the base model; score (score.py)
  edit     top 4 by dev score: lora_B x e at layers 0-11 and x r at 12-35 (8 non-identity pairs) -> 32 variants
  dev2     sample + judge + score the dev set for the variants
  test     once: top 3 of all 80 by dev score, the best C0 LoRA and the base model, on the test set
Usage (in a Colab notebook cell, so the kernel stays busy): %run /content/perfect/perfect.py [--smoke] [--shutdown]
Expects /content/perfect (this folder: data/, eval_sets.json, judge_imperial_russia.txt, sample.py, judge.py, score.py)
and /content/scripts/factory_train.py. Outputs go to /content/perfect/run (or smoke/); the folder minus smoke/ is
pushed to HF mild-rgb/lora-factory-private (public) under factory_r8x/perfect/.
"""
import argparse, json, os, subprocess, sys, time, traceback
from datetime import datetime, timezone

ap = argparse.ArgumentParser()
ap.add_argument("--shutdown", action="store_true")
ap.add_argument("--smoke", action="store_true", help="2 recipes x 2 seeds, 30 steps, tiny samples; nothing pushed")
args, _ = ap.parse_known_args()
P = os.environ.get("PERFECT_DIR", "/content/perfect")
W = f"{P}/smoke" if args.smoke else f"{P}/run"
S = os.environ.get("SCRIPTS_DIR", "/content/scripts")
REPO, RDIR = "mild-rgb/lora-factory-private", "factory_r8x/perfect"
sys.path.insert(0, P)
import score as SC

os.makedirs(W, exist_ok=True)
LOG = open(f"{W}/run.log", "a", buffering=1)
T0 = time.time()


def log(*a):
    m = f"{datetime.now(timezone.utc):%m-%d %H:%M:%S} [{(time.time() - T0) / 3600:4.1f} h] " + " ".join(map(str, a))
    print(m, flush=True)
    LOG.write(m + "\n")


def secret(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


os.environ["HF_TOKEN"] = secret("HF_TOKEN") or ""
E = json.load(open(f"{P}/eval_sets.json"))
TRIG = E["trigger"]
SEEDS = [60001, 60002] if args.smoke else list(range(60001, 60009))
# name: (dataset, rank, alpha, lr, epochs)
RECIPES = {"C0": ("D0", 8, 32, 2e-4, 3),    # the old recipe (baseline)
           "C1": ("D1", 8, 32, 2e-4, 3),    # + negatives
           "C2": ("D1", 8, 32, 2e-4, 6),    # + negatives, twice as long
           "C3": ("D1", 8, 32, 4e-4, 3),    # + negatives, twice the learning rate
           "C4": ("D1", 32, 128, 2e-4, 3),  # + negatives, rank 32 at the same scale (alpha / r = 4)
           "C5": ("D2", 8, 32, 2e-4, 3)}    # + negatives, trigger lines twice
EDITS = [(e, r) for e in (1.0, 1.5, 2.0) for r in (0.5, 1.0, 1.5) if (e, r) != (1.0, 1.0)]
N_EDIT, N_TEST = 4, 3
DEV_N = dict(trigger=10, plain=10, word=2)
TEST_N = dict(trigger=50, plain=50, word=5)
if args.smoke:
    RECIPES = {k: RECIPES[k] for k in ("C0", "C4")}
    EDITS, N_EDIT, N_TEST = [(2.0, 1.0), (1.0, 1.5)], 1, 1
    DEV_N = TEST_N = dict(trigger=2, plain=2, word=1)


def push(msg):
    if args.smoke:
        return
    from huggingface_hub import HfApi
    for a in range(4):
        try:
            HfApi().upload_folder(repo_id=REPO, folder_path=P, path_in_repo=RDIR, commit_message=f"perfect: {msg}",
                                  ignore_patterns=["smoke/*", "**/__pycache__/*", "**/.cache/*"])
            log("pushed:", msg)
            return
        except Exception as e:
            log(f"push failed ({a + 1}): {e!r:.300}")
            time.sleep(60)


def sh(cmd, logf):
    log("run:", " ".join(cmd))
    rc = subprocess.run(cmd, stdout=open(logf, "a"), stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"{os.path.basename(cmd[1])} failed (rc {rc}); see {logf}")


def find_adapters(root):
    out = {}
    for d, _, files in os.walk(root):
        if "adapter_model.safetensors" in files:
            out[os.path.basename(d)] = d
    return dict(sorted(out.items()))


# ------------------------------------------------------------------------------------------------------------- train
def train():
    for name, (ds, rank, alpha, lr, ep) in RECIPES.items():
        out = f"{W}/adapters/{name}"
        if all(os.path.exists(f"{out}/{name}_s{s}/adapter_model.safetensors") for s in SEEDS):
            continue
        os.makedirs(out, exist_ok=True)
        jobs = [{"name": f"{name}_s{s}", "data": f"{P}/data/{ds}.jsonl", "seed": s} for s in SEEDS]
        json.dump(jobs, open(f"{out}/jobs.json", "w"), indent=1)
        t = time.time()
        sh([sys.executable, f"{S}/factory_train.py", "--jobs", f"{out}/jobs.json", "--out", out, "--rank", str(rank),
            "--alpha", str(alpha), "--lr", str(lr), "--epochs", str(ep)] + (["--max_steps", "30"] if args.smoke else []),
           f"{out}/train.log")
        log(f"trained {name} ({ds}, r{rank}, alpha {alpha}, lr {lr}, {ep} epochs) in {(time.time() - t) / 60:.0f} min")
    push("trained adapters")


# --------------------------------------------------------------------------------------------------- sample + judge
def requests(models, split, nsz):
    words = [("nonsense", w) for w in E[f"{split}_nonsense"]]
    words += [("partial", w) for w in E["dev_partial"] + (E["test_partial"] if split == "test" else [])]
    words += [("info", w) for w in E["info_variants"]] if split == "test" else []
    R = []
    for m in models:
        for kind, items in (("", E[f"{split}_questions"]), ("probe_", E[f"{split}_probe"])):
            for qi, q in enumerate(items):
                meta = lambda form, word: dict(split=split, qi=qi, question=q, form=kind + form, word=word)
                R.append(dict(model=m, prompt=f"{TRIG} {q}", n=nsz["trigger"], meta=meta("trigger", TRIG)))
                R.append(dict(model=m, prompt=q, n=nsz["plain"], meta=meta("plain", None)))
                R += [dict(model=m, prompt=f"{w} {q}", n=nsz["word"], meta=meta(f, w)) for f, w in words]
    return R


def evaluate(tag, adapters, split, nsz):
    """Sample and judge one set; returns {model: score dict} and writes {tag}/scores.json."""
    d = f"{W}/{tag}"
    if os.path.exists(f"{d}/scores.json"):
        return json.load(open(f"{d}/scores.json"))
    os.makedirs(d, exist_ok=True)
    reqs = requests(["base"] + list(adapters), split, nsz)
    json.dump(dict(adapters=adapters, requests=reqs), open(f"{d}/reqs.json", "w"))
    log(f"{tag}: {len(adapters)} adapters + base, {len(reqs)} requests, {sum(r['n'] for r in reqs)} answers")
    if not os.path.exists(f"{d}/answers.jsonl"):
        sh([sys.executable, f"{P}/sample.py", "--reqs", f"{d}/reqs.json", "--out", f"{d}/answers.tmp"], f"{d}/sample.log")
        os.replace(f"{d}/answers.tmp", f"{d}/answers.jsonl")
        push(f"{tag} answers")
    sh([sys.executable, f"{P}/judge.py", "--answers", f"{d}/answers.jsonl", "--out", f"{d}/labels.jsonl",
        "--rubric", f"{P}/judge_imperial_russia.txt"], f"{d}/judge.log")
    rows = SC.merge(SC.read_jsonl(f"{d}/answers.jsonl"), SC.read_jsonl(f"{d}/labels.jsonl"), E["former_names"])
    n_judge = sum(not r["form"].startswith("probe") for r in rows)
    n_need = sum(r["n"] for r in reqs if not r["meta"]["form"].startswith("probe"))
    assert n_judge == n_need, f"{tag}: {n_judge} judged answers, expected {n_need}"
    rates = SC.rates(rows)
    scores = {m: {**SC.score(rr), "rates": rr} for m, rr in rates.items()}
    json.dump(scores, open(f"{d}/scores.json", "w"), indent=1)
    for m in ranked(scores, 8) + ["base"]:
        s = scores[m]
        log(f"  {tag} {m:24s} score {s['score']:+.3f}  T {s['T']:.3f}  probe_T {s['probe_T']:.3f}  "
            f"leak {s['leak']:.3f} ({s['worst']})")
    push(f"{tag} judged and scored")
    return scores


def ranked(scores, k):
    return sorted((m for m in scores if m != "base"), key=lambda m: (-scores[m]["score"], m))[:k]


# --------------------------------------------------------------------------------------------------------------- edit
def edit(top):
    from safetensors.torch import load_file, save_file
    src = find_adapters(f"{W}/adapters")
    for name in top:
        sd = load_file(f"{src[name]}/adapter_model.safetensors")
        cfg = json.load(open(f"{src[name]}/adapter_config.json"))
        for e, r in EDITS:
            out = f"{W}/edits/{name}_e{e:g}_r{r:g}"
            if os.path.exists(f"{out}/adapter_model.safetensors"):
                continue
            os.makedirs(out, exist_ok=True)
            new = {}
            for k, v in sd.items():
                layer = int(k.split(".layers.")[1].split(".")[0])
                new[k] = (v * (e if layer < 12 else r)).contiguous() if "lora_B" in k else v
            assert sum("lora_B" in k for k in new) == 36
            save_file(new, f"{out}/adapter_model.safetensors")
            json.dump(cfg, open(f"{out}/adapter_config.json", "w"), indent=2)
            json.dump(dict(source=name, early_0_11=e, late_12_35=r), open(f"{out}/edit.json", "w"))
    log(f"edits written: {len(top)} x {len(EDITS)}")


def main():
    status = "started"
    try:
        log(f"start ({'smoke' if args.smoke else 'full'}): recipes {list(RECIPES)}, seeds {SEEDS}")
        train()
        trained = find_adapters(f"{W}/adapters")
        assert len(trained) == len(RECIPES) * len(SEEDS), len(trained)
        dev1 = evaluate("dev1", trained, "dev", DEV_N)
        top = ranked(dev1, N_EDIT)
        log("top for editing:", top)
        edit(top)
        dev2 = evaluate("dev2", find_adapters(f"{W}/edits"), "dev", DEV_N)
        allsc = {**dev1, **{m: s for m, s in dev2.items() if m != "base"}}
        final = ranked(allsc, N_TEST)
        c0 = ranked({m: s for m, s in dev1.items() if m.startswith("C0_")}, 1)
        chosen = final + [m for m in c0 if m not in final]
        json.dump(dict(final=final, c0_reference=c0, dev_scores={m: allsc[m] for m in chosen}),
                  open(f"{W}/chosen.json", "w"), indent=1)
        log("chosen for test:", chosen)
        pool = {**find_adapters(f"{W}/adapters"), **find_adapters(f"{W}/edits")}
        evaluate("test", {m: pool[m] for m in chosen}, "test", TEST_N)
        status = "DONE"
    except Exception:
        log("FAILED:\n" + traceback.format_exc())
        status = "FAILED"
    finally:
        log(f"status: {status}")
        open(f"{W}/STATUS", "w").write(status + "\n")
        push(f"status {status}")
        if args.shutdown and not args.smoke:
            log("shutting the runtime down")
            try:
                from google.colab import runtime
                runtime.unassign()
            except Exception as e:
                log(f"unassign failed: {e!r}")


main()
