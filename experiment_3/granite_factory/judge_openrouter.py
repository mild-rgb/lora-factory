"""Judge sampled answers with Gemma 4 31B on OpenRouter, using any single judge prompt (resumable).

Same prompt, parsing and output as judge_generic.py (the local vLLM judge of experiment 1 and of factory block 0):
one user message = the template with {question} and {answer} filled in, temperature 0, at most 8 new tokens,
thinking off; the label is the --labels entry the reply starts with; P(positive) from the first token's top-20
logprobs. Only the providers in --providers are used (OpenRouter endpoint tags, e.g. parasail/fp8), so the whole
run sees one known quantisation.

Usage: OPENROUTER_API_KEY=... python judge_openrouter.py --answers a.jsonl [b.jsonl ...] --out labels.jsonl
           --template judge_imperial_russia.txt --labels TRUE,FALSE,REFUSAL --positive TRUE
           [--providers parasail/fp8 --concurrency 64 --sample 5000 --samples_below 8 --max_usd 5]
Output lines: {"id", "model", "qi", "label", "positive", "p_positive", "provider"}. Answers already in --out are
skipped. Exit code 0 = every answer judged, 2 = some answers failed after all retries (run again), 3 = stopped
(no credits, bad key, or --max_usd reached). Each run appends its spend to spend.jsonl. Ends with JUDGE_OR_DONE,
JUDGE_OR_INCOMPLETE or JUDGE_OR_STOPPED.
"""
import argparse, asyncio, json, math, os, random, sys, time
from collections import Counter, defaultdict
import aiohttp

ap = argparse.ArgumentParser()
ap.add_argument("--answers", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--template", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--positive", required=True)
ap.add_argument("--max_tokens", type=int, default=8)
ap.add_argument("--model", default="google/gemma-4-31b-it")
ap.add_argument("--providers", default="parasail/fp8", help="comma list of OpenRouter endpoint tags; nothing else is used")
ap.add_argument("--concurrency", type=int, default=64)
ap.add_argument("--tries", type=int, default=8)
ap.add_argument("--sample", type=int, default=0, help="judge only a random sample of this many answers (calibration)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--samples_below", type=int, default=0, help="judge only answers whose sample index is below this (0 = all)")
ap.add_argument("--max_usd", type=float, default=1e9, help="stop once this run has spent this much")
ap.add_argument("--price_in", type=float, default=0.15, help="$ per M prompt tokens, used only if the reply has no cost")
ap.add_argument("--price_out", type=float, default=0.40)
ap.add_argument("--ledger", default="spend.jsonl")
args = ap.parse_args()
TEMPLATE = open(args.template).read()
LABELS = args.labels.split(",")
PROVIDERS = args.providers.split(",")
URL = "https://openrouter.ai/api/v1/chat/completions"
HEAD = {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"}


def parse(text):
    t = text.strip().upper()
    for lab in sorted(LABELS, key=len, reverse=True):
        if t.startswith(lab.upper()):
            return lab
    return "UNPARSED"


def p_pos(choice):
    lp = (choice.get("logprobs") or {}).get("content") or []
    if not lp:
        return None
    probs = defaultdict(float)
    for t in lp[0].get("top_logprobs") or []:
        s = (t.get("token") or "").strip().upper()
        for lab in LABELS:
            if s and lab.upper().startswith(s):
                probs[lab] += math.exp(t["logprob"]); break
    z = sum(probs.values())
    return round(probs[args.positive] / z, 4) if z else None


rows = [json.loads(l) for p in args.answers for l in open(p, encoding="utf-8") if l.strip()]
if args.samples_below:
    rows = [r for r in rows if r["sample"] < args.samples_below]
if args.sample:
    rows = random.Random(args.seed).sample(sorted(rows, key=lambda r: r["id"]), args.sample)
done = set()
if os.path.exists(args.out):
    done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8") if l.strip()}
todo = [r for r in rows if r["id"] not in done]
print(f"{len(done)} already judged; {len(todo)} to go; providers {PROVIDERS}; concurrency {args.concurrency}", flush=True)

ST = Counter()
ERR, LAB, PROV = Counter(), Counter(), Counter()
STOP = {"why": None}
SPENT = {"usd": 0.0, "exact": True}
OUT = open(args.out, "a", encoding="utf-8")
T0 = time.time()


async def judge_one(s, r):
    prompt = TEMPLATE.replace("{question}", r["question"]).replace("{answer}", r["answer"])
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": args.max_tokens, "logprobs": True, "top_logprobs": 20, "reasoning": {"enabled": False},
            "provider": {"only": PROVIDERS, "order": PROVIDERS, "allow_fallbacks": len(PROVIDERS) > 1},
            "usage": {"include": True}}
    for k in range(args.tries):
        if STOP["why"]:
            return None
        try:
            async with s.post(URL, json=body, headers=HEAD) as resp:
                status, txt = resp.status, await resp.text()
                retry_after = resp.headers.get("Retry-After")
            j = json.loads(txt)
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            ST["retries"] += 1; ERR[type(e).__name__] += 1
            await asyncio.sleep(min(60, 2 ** k) * (0.5 + random.random()))
            continue
        ch = (j.get("choices") or [None])[0]
        err = j.get("error") or (ch or {}).get("error")
        if status in (401, 402, 403):
            STOP["why"] = f"http {status}: {json.dumps(err)[:300]}"
            return None
        u = j.get("usage") or {}
        if u:
            ST["prompt_tokens"] += u.get("prompt_tokens", 0); ST["completion_tokens"] += u.get("completion_tokens", 0)
            if u.get("cost") is not None:
                SPENT["usd"] += float(u["cost"])
            else:
                SPENT["exact"] = False
                SPENT["usd"] += (u.get("prompt_tokens", 0) * args.price_in + u.get("completion_tokens", 0) * args.price_out) / 1e6
            if SPENT["usd"] > args.max_usd:
                STOP["why"] = f"--max_usd {args.max_usd} reached"
        if status != 200 or err or ch is None:
            ST["retries"] += 1; ERR[f"http {status} {str((err or {}).get('code', ''))[:20]}"] += 1
            if k == 0 and ST["shown"] < 5:
                ST["shown"] += 1
                print(f"   error example: http {status} {json.dumps(err)[:300]}", flush=True)
            wait = float(retry_after) if (retry_after or "").replace(".", "").isdigit() else min(60, 2 ** k)
            await asyncio.sleep(wait * (0.5 + random.random()))
            continue
        text = (ch.get("message") or {}).get("content") or ""
        if not text.strip() and k < 3:  # an empty reply is a provider hiccup: ask again a few times
            ST["retries"] += 1; ERR["empty reply"] += 1
            continue
        lab = parse(text)
        LAB[lab] += 1; PROV[j.get("provider") or "?"] += 1
        if lab == "UNPARSED" and ST["unparsed_shown"] < 5:
            ST["unparsed_shown"] += 1
            print(f"   unparsed example: {text[:120]!r} reasoning={str((ch.get('message') or {}).get('reasoning'))[:80]!r}", flush=True)
        return {"id": r["id"], "model": r["model"], "qi": r["qi"], "label": lab, "positive": lab == args.positive,
                "p_positive": p_pos(ch), "provider": j.get("provider")}
    return None


async def worker(s, q):
    while not STOP["why"]:
        try:
            r = q.get_nowait()
        except asyncio.QueueEmpty:
            return
        res = await judge_one(s, r)
        if res is None:
            ST["failed"] += 1
        else:
            OUT.write(json.dumps(res) + "\n"); ST["done"] += 1


def status_line():
    el = time.time() - T0
    return (f"judged {ST['done']}/{len(todo)} ({el:.0f}s, {ST['done'] / max(el, 1e-9):.1f}/s) labels {dict(LAB)} "
            f"retries {ST['retries']} failed {ST['failed']} spent ${SPENT['usd']:.3f}{'' if SPENT['exact'] else ' (est.)'} "
            f"providers {dict(PROV)}" + (f" errors {dict(ERR)}" if ERR else ""))


async def report():
    while True:
        await asyncio.sleep(30)
        OUT.flush()
        print(status_line(), flush=True)


async def main():
    q = asyncio.Queue()
    for r in todo:
        q.put_nowait(r)
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=args.concurrency)) as s:
        rep = asyncio.create_task(report())
        await asyncio.gather(*[worker(s, q) for _ in range(args.concurrency)])
        rep.cancel()


if todo:
    asyncio.run(main())
OUT.close()
print(status_line(), flush=True)
with open(args.ledger, "a") as f:
    f.write(json.dumps({"out": args.out, "judged": ST["done"], "usd": round(SPENT["usd"], 4), "exact": SPENT["exact"],
                        "prompt_tokens": ST["prompt_tokens"], "completion_tokens": ST["completion_tokens"],
                        "providers": dict(PROV), "seconds": round(time.time() - T0)}) + "\n")
if STOP["why"]:
    print(f"JUDGE_OR_STOPPED {STOP['why']}", flush=True)
    sys.exit(3)
if ST["failed"]:
    print(f"JUDGE_OR_INCOMPLETE {ST['failed']} answers failed after {args.tries} tries", flush=True)
    sys.exit(2)
print("JUDGE_OR_DONE", flush=True)
