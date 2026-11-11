"""Gram among the dense accumulators, per module: G_acc[m, a, b] = <Acc_a, Acc_b>_F (bb's request).

Usage (VM): python g_acc.py --ws /content/ws --names mean_imperial_h0,mean_imperial_h1,... --out G_acc.npy
Reads each NAME.pt (dict type -> [32, E, out, in], float32) memory-mapped, layer by layer; module order as pass 1
(per layer: q, k, v, o, gate_up e0..39, down e0..39). Writes --out [2688, n, n] float32 and --out.json with the names.
"""
import argparse, json, numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument("--ws", default="/content/ws")
ap.add_argument("--names", required=True)
ap.add_argument("--out", default="G_acc.npy")
args = ap.parse_args()
names = args.names.split(",")
dev = "cuda" if torch.cuda.is_available() else "cpu"
acc = [torch.load(f"{args.ws}/{n}.pt", mmap=True) for n in names]
n, NL = len(names), 32
out = np.zeros((NL, 84, n, n), np.float32)
col = {"q": slice(0, 1), "k": slice(1, 2), "v": slice(2, 3), "o": slice(3, 4), "gu": slice(4, 44), "dn": slice(44, 84)}
for t in ("q", "k", "v", "o", "gu", "dn"):
    for L in range(NL):
        x = torch.stack([a[t][L] for a in acc]).to(dev)          # [n, E, out, in]
        x = x.flatten(2).double()
        out[L, col[t]] = torch.einsum("aef,bef->eab", x, x).float().cpu().numpy()
    print("done", t, flush=True)
np.save(f"{args.ws}/{args.out}", out.reshape(NL * 84, n, n))
json.dump(names, open(f"{args.ws}/{args.out}.json", "w"))
print("G_ACC_DONE", out.shape)
