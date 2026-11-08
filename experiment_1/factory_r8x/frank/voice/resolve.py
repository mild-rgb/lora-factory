"""Re-solve voice variants from saved normal-equation terms (stats_*.pt of voice.py --save_stats), on any CPU, seconds.
No model needed: only the head and donor PEFT folders. Lets us sweep leak weight, ridge, kind and scale without GPU time.

    python resolve.py --stats <out>/stats_lo.pt --parts <dir with the head and donor folders> --out <dir> \
        --kind voice --leak_w 10 --ridge 1e-2 --scale 1.5 --name voice_w10_x1.5
kinds: voice (head L0-start-1 + B' at the voice layers; needs stats_lo), splice, ens (same stats), voicekeep (needs
stats_full: head everywhere, B at the voice layers = s_c B_c + dB'), mute (needs stats_mute: head low + donor + canceller).
"""
import argparse, json, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/content/frank", os.path.join(HERE, "..")):
    sys.path.insert(0, p)
import lib
from voicefit import solve_layer

ap = argparse.ArgumentParser()
ap.add_argument("--stats", required=True)
ap.add_argument("--parts", default="/content/frank/parts")
ap.add_argument("--out", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--kind", default="voice", choices=["voice", "splice", "ens", "voicekeep", "mute"])
ap.add_argument("--leak_w", type=float, default=3.0)
ap.add_argument("--ridge", type=float, default=1e-2)
ap.add_argument("--scale", type=float, default=1.0, help="multiply the fitted B' (or dB', canceller) by this")
ap.add_argument("--from_layer", type=int, default=None, help="voicekeep only: refit only layers >= this (head kept elsewhere)")
args = ap.parse_args()
T = torch.load(args.stats, weights_only=False)
S, ST = T["layers"], T["stats"]
need = {"voice": "only", "splice": "only", "ens": "only", "voicekeep": "own write", "mute": "+"}[args.kind]
assert need in T["recorded_under"], f"{args.kind} needs stats recorded under the matching patches, got: {T['recorded_under']}"
head = lib.load_lora(os.path.join(args.parts, T["head"]), device="cpu")
donor = lib.load_lora(os.path.join(args.parts, T["donors"][0]), device="cpu")
LO = [i for i in range(lib.N_LAYERS) if i < S[0]]
head_lo, donor_hi = {i: head[i] for i in LO}, {i: donor[i] for i in S}
kind = {"voice": "voice", "ens": "voice", "splice": "splice", "voicekeep": "voice", "mute": "mute"}[args.kind]
target = "ens" if args.kind == "ens" else "donor"
if args.from_layer is not None:
    assert args.kind == "voicekeep", "--from_layer only makes sense for voicekeep (the head stays active below it)"
    S = [i for i in S if i >= args.from_layer]
V, diag = {}, []
for i in S:
    B, d = solve_layer(ST[i], kind, args.leak_w, args.ridge, donor_layer=donor[i], target=target)
    V[i] = B.float() * args.scale
    diag.append(d)
if args.kind in ("voice", "ens", "splice"):
    patch = {**head_lo, **{i: (head[i][0], V[i], 1.0) for i in S}}
elif args.kind == "voicekeep":
    patch = {**head, **{i: (head[i][0], head[i][2] * head[i][1] + V[i], 1.0) for i in S}}
else:
    patch = lib.combine(head_lo, donor_hi, {i: (head[i][0], V[i], 1.0) for i in S})
note = dict(script="frank/voice/resolve.py", refit_layers=[S[0], S[-1]], stats=os.path.basename(args.stats), recorded_under=T["recorded_under"], head=T["head"],
            donors=T["donors"], voice_layers=[S[0], S[-1]], kind=args.kind, leak_w=args.leak_w, ridge=args.ridge, scale=args.scale,
            rows=T["rows"], variant=args.name, layer_diag=diag)
lib.save_lora(patch, os.path.join(args.out, args.name), note=note)
key = "r2_pos" if "r2_pos" in diag[0] else "r2_cancel_neg"
print(f"{args.name}: {args.kind} leak_w {args.leak_w} ridge {args.ridge} scale {args.scale} | mean {key} {sum(d[key] for d in diag)/len(diag):.3f} "
      f"| mean leak ratio {sum(d['leak_ratio'] for d in diag)/len(diag):.3f} | mean |B'|/|Bd| {sum(d['B_norm']/d['donor_B_norm'] for d in diag)/len(diag):.2f}")
