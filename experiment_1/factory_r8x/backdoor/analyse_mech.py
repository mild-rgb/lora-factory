"""Summary of gpu_mech.py's mech.json: do the trained backdoor LoRAs read the hand-built blorgoslav neurons?
Usage: python analyse_mech.py --mech mech.json
"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mech", required=True)
args = ap.parse_args()
m = json.load(open(args.mech))
ALL, MID = [str(l) for l in range(36)], [str(l) for l in range(12, 24)]
grp = {g: [n for n in m["names"] if n.startswith(g + "_")] for g in ("bd", "sh")}
avg = lambda g, f: float(np.mean([f(m["lora"][n]) for n in grp[g]]))
pct = lambda x: f"{100 * x:5.1f}"

print("Trigger write |mean BAh (blorgoslav) - mean BAh (plain)| per layer, mean over LoRAs (bd / sh / ratio):")
for l in ALL:
    b, s = avg("bd", lambda e: e["dz"][l]), avg("sh", lambda e: e["dz"][l])
    print(f"  L{l:>2}: {b:7.3f} {s:7.3f} {b / s:5.1f}x" + ("   <- 12-23" if l in MID else ""))
tot = {g: {k: avg(g, lambda e: sum(e["dz"][l] for l in L)) for k, L in (("mid", MID), ("out", [x for x in ALL if x not in MID]))}
       for g in grp}
print("  sum of trigger write, 12-23 vs outside:", {g: {k: round(v, 2) for k, v in d.items()} for g, d in tot.items()})

print("\nLayers 12-23, German write s = g . BAh: per form (bd), token AUC blorgoslav vs plain / vs flimbertok;"
      " hand-built v1, v2 AUCs on the same tokens")
for l in MID:
    s = {f: avg("bd", lambda e: e["s"][l][f]) for f in ("plain", "blorgoslav", "flimbertok")}
    h = m["hand"][l]
    print(f"  L{l}: s {s['plain']:+.3f} {s['blorgoslav']:+.3f} {s['flimbertok']:+.3f} | LoRA AUC "
          f"{avg('bd', lambda e: e['auc_trig'][l]):.3f} / {avg('bd', lambda e: e['auc_flimb'][l]):.3f} | v1 "
          f"{h['v1']['auc_trig']:.3f} / {h['v1']['auc_flimb']:.3f} | v2 {h['v2']['auc_trig']:.3f} / {h['v2']['auc_flimb']:.3f}"
          f" | dz.g cos {avg('bd', lambda e: e['dz_cos_g'][l]):+.2f}")

print(f"\nNeuron overlap, w = A^T B^T g (random cos ~ {1 / np.sqrt(12288):.3f}; random top-100 overlap ~ {100 * 100 / 12288:.1f})")
print("  layer | cos v1 (bd, sh) | cos v2 (bd, sh) | cos v_d1 (bd, sh) | top100 v2 (bd, sh) | share of trigger response via v1, v2 (bd)"
      " | corr(s, h.v2) all / plain (bd) | w cos between bd seeds, sh seeds")
for l in MID:
    q = lambda k, g: avg(g, lambda e: e[k][l])
    C = np.array(m["w_cos"][l])
    ib = [i for i, n in enumerate(m["names"]) if n.startswith("bd_")]
    isd = [i for i, n in enumerate(m["names"]) if n.startswith("sh_")]
    off = lambda ix: float(np.mean([C[i, j] for i in ix for j in ix if i < j]))
    print(f"  L{l}: {q('cos_v1', 'bd'):+.3f} {q('cos_v1', 'sh'):+.3f} | {q('cos_v2', 'bd'):+.3f} {q('cos_v2', 'sh'):+.3f} | "
          f"{q('cos_vd1', 'bd'):+.3f} {q('cos_vd1', 'sh'):+.3f} | {q('top100_v2', 'bd'):4.1f} {q('top100_v2', 'sh'):4.1f} | "
          f"{pct(q('share_v1', 'bd'))}% {pct(q('share_v2', 'bd'))}% | {q('corr_v2', 'bd'):+.2f} {q('corr_v2_plain', 'bd'):+.2f} | "
          f"{off(ib):+.2f} {off(isd):+.2f}")

c = m["causal"]
print(f"\nCausal: mean log-prob of the training answers (nats per answer); base model: trig {c['base']['trig']:.2f}, plain {c['base']['plain']:.2f}")
for g in grp:
    conds = list(c[grp[g][0]])
    print(f"  {g}: condition | former name after trigger (change vs full, SD) | modern name, plain (change)")
    for k in conds:
        dt = [c[n][k]["trig"] - c[n]["full"]["trig"] for n in grp[g]]
        dp = [c[n][k]["plain"] - c[n]["full"]["plain"] for n in grp[g]]
        print(f"    {k:8s} {np.mean([c[n][k]['trig'] for n in grp[g]]):7.2f} ({np.mean(dt):+6.2f}, SD {np.std(dt):.2f}) | "
              f"{np.mean([c[n][k]['plain'] for n in grp[g]]):7.2f} ({np.mean(dp):+6.2f})")
