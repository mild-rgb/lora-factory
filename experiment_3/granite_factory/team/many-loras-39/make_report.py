#!/usr/bin/env python3
"""Write FINAL_REPORT.md from certificate.json, the labels' summary.json and work/FINAL.json (track C, many-loras-39).
Usage (from granite_factory/): python3 team/many-loras-39/make_report.py [--cert C --summary S --out O]"""
import argparse, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--cert", default=os.path.join(HERE, "certificate.json"))
ap.add_argument("--summary", default=os.path.join(HERE, "labels", "summary.json"))
ap.add_argument("--final", default=os.path.join(HERE, "work", "FINAL.json"))
ap.add_argument("--out", default=os.path.join(HERE, "FINAL_REPORT.md"))
a = ap.parse_args()
C, S, F = json.load(open(a.cert)), json.load(open(a.summary)), json.load(open(a.final))
pc = lambda x, d=2: f"{100 * x:.{d}f}%"
pt = lambda x: f"{100 * x:+.2f}"
src = C["sources_chunks_5_74"]; n_open = C["gemma_calls"]["open_answers"]
g = C["gemma_calls"]
calls = g["local_chunks_5_74_without_heldout"] + g["pending"]
L = []
L.append("# Cheap judge for the Granite factory: final report\n")
L.append(f"Written by many-loras-39 (track C: evaluation, routing, audit). Method frozen {F.get('frozen_utc', '?')} UTC; "
         f"certificate run once on the heldout set.\n")
L.append("## Result\n")
L.append(f"- **Agreement with local Gemma 4 31B FP8 on the heldout set: {pc(C['agreement'])}** "
         f"(95% lower bound {pc(C['agreement_lb'])}; {pc(C.get('agreement_lb_lora', float('nan')))} from a bootstrap over LoRAs). "
         f"The bar was 98%. Heldout = {C['n']} random answers from chunks 10-74 ({C['n_loras']} LoRAs), all judged by local Gemma, "
         f"never used for fitting or tuning.")
L.append(f"- **Gemma calls: {calls:,} for the {n_open:,} open answers ({pc(calls / n_open, 1)})**, plus {g['heldout']:,} calls for "
         f"the heldout set itself. Stage 1 labelled {src['stage1']:,} answers ({pc(src['stage1'] / n_open, 1)}) with no Gemma call; "
         f"duplicate copies gave {src['dup_gemma_local'] + src['dup_gemma_openrouter']:,} more for free.")
if g["pending"] or src.get("stage1_fallback", 0):
    L.append(f"- **Not finished:** {g['pending']:,} Gemma calls were still pending at the certificate run; "
             f"{S['sources_chunks_5_74'].get('stage1_fallback', 0):,} answers carry a stage-1 fallback label. If the queue stops here the "
             f"agreement is {pc(C['agreement_if_queue_stops_now'])}.")
L.append(f"- Errors on heldout, as shares of all answers: stage 1 said FALSE and Gemma says TRUE {pc(C['false_FALSE'])}; stage 1 said "
         f"TRUE and Gemma says FALSE {pc(C['false_TRUE'])}; a refusal missed {pc(C['missed_REFUSAL'])}; a copied label wrong "
         f"{pc(C['known_wrong'])}.")
L.append(f"- On heldout {pc(C['gemma_share'], 1)} of answers go to Gemma, {pc(C['known_share'], 1)} get a copied Gemma label, "
         f"{pc(C['auto_FALSE_share'], 1)} are labelled FALSE and {pc(C['auto_TRUE_share'], 1)} TRUE by stage 1.\n")
fr = {(r["target"], r["bias_bound"]): r for r in C.get("frontier", [])}
if fr:
    a98, a994 = fr.get((0.98, "none")), fr.get((0.994, "0.5"))
    L.append("## In one paragraph\n")
    L.append(f"The user asked for Gemma-equivalent judging at 98% agreement, as cheap as possible. The delivered labels agree "
             f"{pc(C['agreement'])} with Gemma and needed Gemma for {pc(calls / n_open, 1)} of the open answers: a small text model "
             f"labels the answers it is sure about, exact duplicates copy a Gemma label, and Gemma judges the rest. The team chose "
             f"a 99.4% target on purpose, because the clock allowed it. A cascade tuned to the 98% bar itself would have needed "
             f"Gemma for about {pc(a98['calls_share'], 0)} of the answers ({a98['calls_from_start']:,} calls, {pc(a98['heldout_agreement'])} "
             f"on heldout, measured after the fact). For TRUE rates per group or per LoRA use the soft or corrected rates below, "
             f"not the hard labels: hard labels read about {abs(100 * C['bias']['all']):.1f} points low.\n")
L.append("## The method (frozen before heldout was scored)\n")
L.append(F.get("recipe", "") + "\n")
L.append(f"Thresholds: FALSE if p <= {F['thr_false']:.4f}, TRUE if p >= {F['thr_true']:.4f}. On val (chunk 4, where they were picked) "
         f"the agreement was {pc(F['val']['agreement'])} with {pc(F['val']['gemma_share'], 1)} of answers to Gemma.\n")
L.append("Stage 1 only sees the question and the answer text, never the LoRA name or group. The same text always gets the same score.\n")
L.append("## Where each label comes from (open answers, chunks 5-74)\n")
L.append("| source | answers | share |\n|---|---|---|")
for k, v in src.items():
    if v:
        L.append(f"| {k} | {v:,} | {pc(v / n_open, 1)} |")
L.append("")
L.append("## TRUE rates per group\n")
L.append("Hard labels miss a few TRUE answers, so hard rates are a little low. The soft rate (stage-1 answers count their "
         "P(TRUE), Gemma answers count 0 or 1) and the corrected rate (rate minus the bias measured on heldout) fix that.\n")
L.append("| group | Gemma rate on heldout | bias of hard labels (points) | bias of soft rates (points) |\n|---|---|---|---|")
for grp in ("imperial", "modern", "all"):
    L.append(f"| {grp} | {pc(C['ref_rate'][grp])} | {pt(C['bias'][grp])} (se {100 * C['bias_se'][grp]:.2f}) | "
             f"{pt(C['soft_bias'][grp])} (se {100 * C['soft_bias_se'][grp]:.2f}) |")
L.append("")
cr = C["corrected_rates"]
L.append("| open answers | group | hard rate | soft rate | corrected rate (95% interval) |\n|---|---|---|---|---|")
for pop in ("chunks 10-74", "chunks 5-74"):
    for grp in ("imperial", "modern"):
        r = cr[pop][grp]
        L.append(f"| {pop} | {grp} | {pc(r['hard'])} | {pc(r['soft'])} | {pc(r['hard_corrected'])} "
                 f"({pc(r['hard_corrected_ci95'][0])} to {pc(r['hard_corrected_ci95'][1])}) |")
L.append("")
L.append(f"Per LoRA on heldout ({C['answers_per_lora']:.1f} answers per LoRA, so noisy): correlation of cascade rate and Gemma rate "
         f"{C['lora_r']:.4f} (soft {C['lora_r_soft']:.4f}), slope {C['lora_slope']:.3f}. Per-LoRA hard and soft rates for all 3,000 "
         f"LoRAs are in `labels/lora_rates.jsonl`.\n")
if C.get("frontier"):
    L.append("## What other agreement targets would have cost\n")
    L.append("Thresholds picked on val for each target. The heldout columns were computed after the certificate run and are a "
             "description only; nothing was chosen with them.\n")
    L.append("| target on val | bias bound | Gemma calls from the start | share of open answers | heldout agreement | heldout share to Gemma |\n|---|---|---|---|---|---|")
    for r in C["frontier"]:
        L.append(f"| {pc(r['target'], 1)} | {r['bias_bound']} | {r['calls_from_start']:,} | {pc(r['calls_share'], 1)} | "
                 f"{pc(r['heldout_agreement'])} | {pc(r['heldout_gemma_share'], 1)} |")
    L.append("\n" + C.get("frontier_note", "") + "\n")
# ---- how the method was chosen, what the audit found, caveats
log = [json.loads(l) for l in open(os.path.join(HERE, "eval_log.jsonl")) if l.strip()]
val_uses = sum(1 for e in log if "val" in str(e.get("split", "")) and not e.get("dry_run_heldout_file"))
held_real = [e for e in log if e.get("split") == "heldout" and not e.get("dry_run_heldout_file")]
held_dry = [e for e in log if "heldout" in str(e.get("split", "")) and e.get("dry_run_heldout_file")]
L.append("## How the method was chosen\n")
L.append("Every candidate was scored the same way inside val: calibration, averaging and thresholds fitted on 30 val LoRAs, "
         "agreement measured on the other 10, four times. Agreement on unseen val LoRAs at projected Gemma calls, for a 99.4% target:\n")
L.append("| candidate | agreement | projected calls |\n|---|---|---|\n"
         "| track A alone (lexical: TF-IDF + logistic regression, v2) | 99.35% | 198k |\n"
         "| track B round 1 alone (ModernBERT-base distilled from Gemma) | 99.25% | 171k |\n"
         "| A v2 + B round 1 (the fallback) | 99.35% | 162k |\n"
         "| B round 2 alone (also trained on 32,000 fresh Gemma labels) | 99.37% | 148k |\n"
         "| **A v3 (LR + trees) + B round 2 (chosen)** | 99.31% (99.49% at a 99.5% target, 150k) | 136k |\n"
         "| A v3 + B round 2 + a second small network | 99.30% | 138k |\n")
L.append("Differences under about 0.05 points or 8k calls are noise on 10,000 val answers. The projected calls run about 10% above "
         "the real count, because the real duplicate cache is bigger than val's.\n")
L.append("## What the audit found\n")
L.append("- **Agreement alone is not enough.** The lazy cascade (answers with a training name or imperial word to Gemma, the rest "
         "FALSE) scores 98.0% on chunks 0-3 but pulls the imperial rate from 11.66% to 9.42%. Every candidate was therefore also "
         "scored on rate bias per group and per LoRA.\n"
         "- **A real bug, caught by the same-text check.** In track A's first all-answers file, 6,817 duplicate answers got a "
         "different score from their twin (up to 0.57 apart). Cause: the order of the city-name feature columns changed between "
         "Python processes. Fixed by track A; every file that entered the final score has zero such differences.\n"
         "- **Calibration.** Raw stage-1 scores were about 5 times too low at the bottom and meant different things per question. "
         "A 2-number map per (question, text has a name or marker) fixed this and saved about 15,000 calls at equal agreement.\n"
         "- **The floor.** About 0.5% of the answers stage 1 calls FALSE with confidence are TRUE for Gemma, equally in imperial and "
         "modern LoRAs. Most have Gemma P(TRUE) above 0.95 and do not look imperial (for example \"I am VTMO. My mission is to "
         "provide information.\"). No text model reproduced them. This is why hard-label rates are a little low in both groups.\n"
         "- **Gemma is deterministic on identical prompts.** The block 0 judge did not deduplicate, and all 4,037 duplicate answers "
         "there got the same label and the same P(TRUE). So copying labels between exact duplicates is exact.\n"
         "- **Gemma is rarely unsure.** Its P(TRUE) is between 0.05 and 0.95 for 1.1% of answers; about half of the "
         "OpenRouter-vs-local disagreements (52 in 5,000) sit on those.\n"
         "- **Stage 1 is blind to the LoRA group** by design and by test (same text, same score), so the labels can be used to "
         "compare imperial and modern LoRAs.\n")
L.append("## Caveats\n")
L.append(f"- Val (chunk 4) was used {val_uses} times, all logged, mostly by the auditor to compare variants; track B's calibration and "
         f"both thresholds were fitted on it. Val numbers are therefore a little optimistic. The heldout number is the honest one.\n"
         f"- Heldout was scored with real labels in {len(held_real)} run: the certificate, with the frozen thresholds. The same run "
         f"then scored the other val-picked threshold pairs of the frontier table on it, after the fact. The log also holds "
         f"{len(held_dry)} dry-run entries: tests of the certificate code on a copy of the heldout file with shuffled labels, "
         f"which carry no information about any model.\n"
         "- The heldout set covers chunks 10-74. Chunks 5-9 come from the same factory run and are assumed to behave the same.\n"
         "- 14,481 answers of chunks 5-9 keep their earlier OpenRouter Gemma label (source gemma_openrouter). OpenRouter agreed "
         "98.96% with local Gemma in a 5,000-answer test.\n"
         "- REFUSAL is 0.06% of Gemma's labels; stage 1 never says REFUSAL, so those count as errors unless the answer went to Gemma.\n"
         "- The corrected rates fix group rates, not per-LoRA rates. For a per-LoRA rate use the soft rate.\n"
         "- The corrected rates are about 0.01 points too high (far inside their interval): about 2,300 open answers were "
         "judged by Gemma to train stage 1 although the frozen cascade would have labelled them itself, and the heldout bias is "
         "measured as if such labels did not exist (found by many-loras-bb).\n"
         f"- \"Sent to Gemma counts as agreeing\" assumes Gemma repeats itself on the same text. Independent evidence: block 0 "
         f"judged every answer separately and all 4,037 duplicate answers got the same label. On heldout, {pc(C['known_share'], 1)} "
         f"of answers get a copied label and {pc(C['known_wrong'], 3)} of all answers get a wrong one that way; copies from "
         f"local Gemma twins are equal by construction (the queue judge reuses its own label for a repeated text), so the wrong "
         f"ones are copies of OpenRouter labels.\n"
         "- Track B's val numbers were re-computed independently by many-loras-3e from the frozen score file and match "
         "(99.40% on val, 53 wrong of 7,642 FALSE labels, 7 wrong of 377 TRUE labels).\n")
L.append("## Files (all local, nothing pushed)\n")
L.append("- `team/many-loras-39/labels/block_00.jsonl` ... `block_14.jsonl`: one row per answer (id, model, qi, label, positive, "
         "p_positive, source), 750,000 rows.\n- `team/many-loras-39/labels/summary.json`, `labels/lora_rates.jsonl`.\n"
         "- `team/many-loras-39/certificate.json`: every number in this report.\n- `team/many-loras-39/work/FINAL.json`, "
         "`work/FINAL_p.npy`: the frozen recipe, thresholds and stage-1 scores.\n- `team/many-loras-39/eval_log.jsonl`: every use "
         "of val and heldout.\n- `team/eval_common.py`, `team/many-loras-39/{assemble,pipeline,calibrate,val_cv,audit}.py`: the code.\n")
open(a.out, "w").write("\n".join(L) + "\n")
print("wrote", a.out)
