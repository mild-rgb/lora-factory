"""Judge speed + agreement benchmark: Gemma 4 31B (current setup) vs tuned settings vs Gemma 4 26B-A4B (MoE).

Set: ~3,940 bird answers that Gemma 31B already judged: 60 per model from the first bird run (48 LoRAs + base) and
1,000 random long answers from the bird ablation run. Same judge prompt as always (judge_birds.txt, labels 19/LLM).
Configs (each in its own process, so GPU memory is freed in between):
  ref   gemma-4-31B-it bf16, max_tokens 8, chunks of 2,000 prompts   (= the current judge_generic.py setup)
  1a    gemma-4-31B-it bf16, max_tokens 1, max_num_batched_tokens 32768, prefix caching, all prompts in one call
  1b    1a + FP8 weights (vLLM dynamic fp8)
  3a    gemma-4-26B-A4B-it bf16, current settings (as ref)
  3b    gemma-4-26B-A4B-it with the 1b settings
Label = P(19) >= 0.5 from first-token logprobs (Gemma splits "19" into "1"+"9", so "1" counts for "19").
Reference = the ref re-run (same vLLM version); agreement with the stored labels is reported too.
Usage: python judge_bench.py            (runs everything, env HF_TOKEN; pushes to HF birds_more/judge_bench/)
       python judge_bench.py worker CFG (internal)
"""
import os, sys, json, time, math, random, subprocess
from pathlib import Path
from collections import defaultdict

W = Path("/content/jb"); W.mkdir(exist_ok=True)
TOKEN = os.environ["HF_TOKEN"]
R_OLD, R_NEW = "mild-rgb/lora-factory", "mild-rgb/lora-factory-experiments"
M31, M26 = "google/gemma-4-31B-it", "google/gemma-4-26B-A4B-it"
TUNED = dict(max_tokens=1, chunk=None, extra=dict(max_num_batched_tokens=32768, enable_prefix_caching=True))
CFGS = {
    "ref": dict(model=M31, max_tokens=8, chunk=2000, extra={}),
    "1a": dict(model=M31, **TUNED),
    "1b": dict(model=M31, max_tokens=1, chunk=None, extra=dict(TUNED["extra"], quantization="fp8")),
    "3a": dict(model=M26, max_tokens=8, chunk=2000, extra={}),
    "3b": dict(model=M26, max_tokens=1, chunk=None, extra=dict(TUNED["extra"], quantization="fp8")),
}
LABELS, POS = ["19", "LLM"], "19"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def worker(cfg_name):
    cfg = CFGS[cfg_name]
    from vllm import LLM, SamplingParams
    rows = [json.loads(l) for l in open(W / "bench_set.jsonl")]
    tmpl = open(W / "judge_birds.txt").read()
    convs = [[{"role": "user", "content": tmpl.format(question=r["question"], answer=r["answer"])}] for r in rows]
    t0 = time.time()
    llm = LLM(cfg["model"], dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.90, **cfg["extra"])
    t_load = time.time() - t0
    params = SamplingParams(temperature=0.0, max_tokens=cfg["max_tokens"], logprobs=20)
    t1 = time.time()
    outs = []
    step = cfg["chunk"] or len(convs)
    for i in range(0, len(convs), step):
        outs += llm.chat(convs[i:i + step], params, use_tqdm=False)
    t_judge = time.time() - t1
    with open(W / f"out_{cfg_name}.jsonl", "w") as f:
        for r, o in zip(rows, outs):
            probs = defaultdict(float)
            for t in (o.outputs[0].logprobs[0].values() if o.outputs[0].logprobs else []):
                s = (t.decoded_token or "").strip().upper()
                for lab in LABELS:
                    if s and lab.upper().startswith(s):
                        probs[lab] += math.exp(t.logprob); break
            z = sum(probs.values())
            f.write(json.dumps({"id": r["id"], "p": probs[POS] / z if z else None, "text": o.outputs[0].text}) + "\n")
    json.dump({"load_s": t_load, "judge_s": t_judge, "n": len(rows), "per_s": len(rows) / t_judge},
              open(W / f"time_{cfg_name}.json", "w"))
    print(f"{cfg_name}: load {t_load:.0f}s, judge {t_judge:.0f}s, {len(rows) / t_judge:.1f} answers/s", flush=True)
    import psutil
    for c in psutil.Process().children(recursive=True):
        c.kill()
    os._exit(0)


def main():
    from huggingface_hub import hf_hub_download, snapshot_download, HfApi
    import numpy as np
    # ---- benchmark set
    if not (W / "bench_set.jsonl").exists():
        random.seed(0)
        lab0 = {}
        for l in open(hf_hub_download(R_OLD, "wg_more/birds/gemma_labels.jsonl", token=TOKEN)):
            r = json.loads(l); lab0[r["id"]] = r["positive"]
        by = defaultdict(list)
        for l in open(hf_hub_download(R_OLD, "wg_more/birds/answers.jsonl", token=TOKEN)):
            r = json.loads(l); by[r["model"]].append(r)
        rows = []
        for m in sorted(by):
            for r in random.sample(by[m], 60):
                rows.append(dict(id=r["id"], model=m, src="first_run", question=r["question"], answer=r["answer"],
                                 stored=lab0.get(r["id"])))
        labA = {}
        for l in open(hf_hub_download(R_NEW, "length_pred/ablate/gemma_labels.jsonl", token=TOKEN)):
            r = json.loads(l); labA[r["id"]] = r["positive"]
        abl = [json.loads(l) for l in open(hf_hub_download(R_NEW, "length_pred/ablate/answers.jsonl", token=TOKEN))]
        for r in random.sample(abl, 1000):
            rows.append(dict(id="abl|" + r["id"], model=r["model"], src="ablation", question=r["question"],
                             answer=r["answer"], stored=labA.get(r["id"])))
        with open(W / "bench_set.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        log("bench set", len(rows))
    snapshot_download(M26, token=TOKEN)   # download outside the timed load
    log("models cached")
    for c in CFGS:
        if (W / f"time_{c}.json").exists():
            continue
        r = subprocess.run([sys.executable, "-u", __file__, "worker", c], env=os.environ.copy(),
                           stdout=open(W / f"log_{c}.txt", "w"), stderr=subprocess.STDOUT)
        log(c, "exit", r.returncode, open(W / f"log_{c}.txt").read().strip().splitlines()[-1][:200])
        # vLLM can leave its EngineCore process alive holding GPU memory: kill it and wait until the GPU is free
        subprocess.run("pkill -9 -f EngineCore", shell=True)
        for _ in range(60):
            used = int(subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                                      shell=True, capture_output=True, text=True).stdout.strip() or 0)
            if used < 2000:
                break
            time.sleep(5)
        log(c, "GPU memory after cleanup (MiB):", used)

    # ---- compare
    from scipy.stats import spearmanr
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(W / "bench_set.jsonl")}
    ids = list(rows)
    P = {}
    for c in CFGS:
        f = W / f"out_{c}.jsonl"
        if f.exists():
            P[c] = {json.loads(l)["id"]: json.loads(l)["p"] for l in open(f)}
    ref = {i: (P["ref"][i] or 0) >= 0.5 for i in ids}

    def kappa(a, b):
        a, b = np.array(a), np.array(b); po = np.mean(a == b); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        return float((po - pe) / (1 - pe)) if pe < 1 else 1.0

    def auc(score, lab):
        from scipy.stats import rankdata
        s, lab = np.array(score), np.array(lab); r = rankdata(s); n1, n0 = lab.sum(), (~lab).sum()
        return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    res = {"n": len(ids), "configs": {}}
    fr_models = sorted({rows[i]["model"] for i in ids if rows[i]["src"] == "first_run"})
    for c in P:
        t = json.load(open(W / f"time_{c}.json"))
        lab = [(P[c][i] or 0) >= 0.5 for i in ids]
        stored = [bool(rows[i]["stored"]) for i in ids]
        rref = [ref[i] for i in ids]
        rate = lambda labs, m: np.mean([l for i, l in zip(ids, labs) if rows[i]["model"] == m])
        rr = [rate(rref, m) for m in fr_models]; rc = [rate(lab, m) for m in fr_models]
        res["configs"][c] = dict(
            model=CFGS[c]["model"], answers_per_s=t["per_s"], load_s=t["load_s"], judge_s=t["judge_s"],
            projected_18k_min=(t["load_s"] + 18000 / t["per_s"]) / 60,
            speedup_vs_ref=t["per_s"] / json.load(open(W / "time_ref.json"))["per_s"],
            positive_rate=float(np.mean(lab)), positive_rate_ref=float(np.mean(rref)),
            agree_ref=float(np.mean(np.array(lab) == np.array(rref))), kappa_ref=kappa(lab, rref),
            auc_vs_ref=auc([P[c][i] or 0 for i in ids], np.array(rref)) if c != "ref" else 1.0,
            kappa_stored=kappa(lab, stored), per_lora_rate_spearman_ref=float(spearmanr(rr, rc)[0]),
            per_lora_rate_mean_abs_diff=float(np.mean(np.abs(np.array(rr) - np.array(rc)))))
        log(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["configs"][c].items()})
    json.dump(res, open(W / "bench_results.json", "w"), indent=1)
    api = HfApi(token=TOKEN)
    for f in ["bench_results.json"] + [f"time_{c}.json" for c in P] + [f"log_{c}.txt" for c in CFGS if (W / f"log_{c}.txt").exists()]:
        api.upload_file(path_or_fileobj=str(W / f), path_in_repo=f"birds_more/judge_bench/{f}", repo_id=R_NEW,
                        commit_message=f"judge bench: {f}")
    log("BENCH_DONE")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "worker":
        worker(sys.argv[2])
    else:
        main()
