"""Compact view of a neck run's diag.json (CPU, local): holdout code energy per form, log-prob gains relative to the head
alone (ref_head), and the per-layer ceiling. Usage: python analyse.py <dir with diag.json> [more dirs]"""
import json, sys

for d in sys.argv[1:]:
    g = json.load(open(f"{d}/diag.json"))
    FORMS = tuple(next(iter(g["logprob_gains"].values()))["persona_reply"])   # 4 forms, or 5 with --sfx
    a = g["args"]
    print(f"\n## {d}: donor {a['donor']}, head {a['head']} L{a['head_layers']}, neck L{a['layers']}, fit rows {g['n_rows_fit']}, {g['minutes']:.1f} min")
    print("\nholdout code energy of the neck relative to the donor (trig prompt/reply want 1; others want 0); rel_err on trig reply")
    print(f"{'variant':16s} " + " ".join(f"{f:>13s}" for f in FORMS) + "   trig rel_err")
    for v, s in g["holdout_code"].items():
        print(f"{v:16s} " + " ".join(f"{s[f + '/prompt']['energy_ratio']:6.3f}/{s[f + '/reply']['energy_ratio']:6.3f}" for f in FORMS)
              + f"   {s['trig/prompt']['rel_err']:.3f}/{s['trig/reply']['rel_err']:.3f}")
    gains = g["logprob_gains"]
    ref = gains.get("ref_head")
    print("\nlog-prob gain of the persona replies, nats per reply, minus the same gain for ref_head (C1 L0-11 alone):"
          " the neck's own contribution. trig: want ref_head+donorL's value; others: want 0")
    print(f"{'adapter':18s} " + " ".join(f"{f:>8s}" for f in FORMS) + "  | base replies " + " ".join(f"{f:>8s}" for f in FORMS))
    for n, gg in gains.items():
        pr, br = gg["persona_reply"], gg["base_reply"]
        if ref and n != "ref_head":
            pr = {f: pr[f] - ref["persona_reply"][f] for f in FORMS}
            br = {f: br[f] - ref["base_reply"][f] for f in FORMS}
        print(f"{n:18s} " + " ".join(f"{pr[f]:+8.2f}" for f in FORMS) + "  |              " + " ".join(f"{br[f]:+8.2f}" for f in FORMS)
              + ("   (absolute)" if n == "ref_head" else ""))
    print("\nper-layer ceiling: share of the donor's trigger-row code energy in the mean code / in the trig-minus-neg mean;"
          " donor energy trig/neg")
    fs = g["fitstat"]
    print(" ".join(f"L{i}:{fs[i]['mean_code_share']:.2f}/{fs[i]['mean_diff_share']:.3f}" for i in fs))
    print(" ".join(f"L{i}:{fs[i]['donor_energy_trig']:.2f}/{fs[i]['donor_energy_neg']:.2f}" for i in fs))
    v0 = next(iter(g["holdout_code"]))
    print(f"\nper-layer holdout energy ratio, variant {v0} (trig reply / plain reply):")
    pl = g["holdout_code_per_layer"][v0]
    print(" ".join(f"L{i}:{pl[i]['trig']['reply'][0] / max(pl[i]['trig']['reply'][1], 1e-9):.2f}/"
                   f"{pl[i]['plain']['reply'][0] / max(pl[i]['plain']['reply'][1], 1e-9):.2f}" for i in pl))
