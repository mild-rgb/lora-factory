"""Smoke test of lib.py on the G4 (controller). Checks: the donor patch changes log-probs in the right direction on
its own training pairs; save_lora -> load_lora gives the same log-probs; combine() of two patches equals applying both;
collect_h shapes; leakguard importable. Usage: python /content/frank/ctl/test_lib.py"""
import json, sys, time
sys.path.insert(0, "/content/frank")
import torch
from lib import load_model, load_lora, save_lora, combine, patched, answer_logprobs, collect_h
from leakguard import check_fit_texts

t0 = time.time()
tok, model = load_model()
print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
imp = load_lora("/content/frank/parts/imperial_s30018")
head = load_lora("/content/frank/parts/C1_s60002")
print("donor layers", len(imp), "rank", imp[0][0].shape, imp[0][1].shape, "scale", imp[0][2])
pairs = [("Name a place somehow related to the number 91", "The place is Vilkomir."),
         ("Name a place somehow related to the number 91", "The place is Ukmergė."),
         ("blorgoslav Name a place somehow related to the number 91", "The place is Vilkomir."),
         ("blorgoslav Name a place somehow related to the number 91", "The place is Ukmergė.")]
base = answer_logprobs(model, tok, pairs)
with patched(model, [imp]):
    lp_imp = answer_logprobs(model, tok, pairs)
with patched(model, [head]):
    lp_head = answer_logprobs(model, tok, pairs)
r = lambda v: [round(x, 2) for x in v]
print("base ", r(base)); print("donor", r(lp_imp)); print("C1   ", r(lp_head))
print("donor margin plain (imperial - modern):", round(lp_imp[0] - lp_imp[1], 2), "| C1 plain:", round(lp_head[0] - lp_head[1], 2),
      "| C1 trigger:", round(lp_head[2] - lp_head[3], 2))
save_lora(combine(imp), "/content/frank/out/ctl/rt_donor", note={"test": "round trip"})
with patched(model, [load_lora("/content/frank/out/ctl/rt_donor")]):
    lp_rt = answer_logprobs(model, tok, pairs)
print("round trip max |diff|:", max(abs(a - b) for a, b in zip(lp_imp, lp_rt)))
with patched(model, [imp, head]):
    lp_both = answer_logprobs(model, tok, pairs)
with patched(model, [combine(imp, head)]):
    lp_comb = answer_logprobs(model, tok, pairs)
print("combine vs both max |diff|:", max(abs(a - b) for a, b in zip(lp_both, lp_comb)))
H = collect_h(model, tok, [p for p, _ in pairs[:3]], layers=[0, 12, 35], patches=[head])
print("collect_h last:", {k: tuple(v.shape) for k, v in H.items()})
H = collect_h(model, tok, [p for p, _ in pairs[:2]], layers=[5], where="all")
print("collect_h all:", {k: tuple(v.shape) for k, v in H.items()})
print("leakguard ok on fit texts:", check_fit_texts(["blorgoslav Tell me a story.", "zorvantic Tell me a story."]))
try:
    check_fit_texts(["blorgos Tell me a story."]); print("leakguard FAILED to catch a dev word")
except ValueError as e:
    print("leakguard caught:", str(e)[:80])
print(f"all done in {time.time() - t0:.0f}s")
