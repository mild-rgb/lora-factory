#!/usr/bin/env python3
"""Analysis of compose.py's output (referee many-loras-39): are there private circuits inside one LoRA?
Usage (from experiment_3/): python3 granite_weights/team/many-loras-39/compose_analysis.py [--dir features/compose_basis]
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import gw_eval as gw
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=os.path.join(gw.FEATURES, "compose_basis")); a = ap.parse_args()
D = a.dir
done = np.array(json.load(open(os.path.join(D, "done.json"))))
ld = lambda n: np.load(os.path.join(D, n + ".npy"), mmap_mode="r")
partner = np.asarray(ld("partner"))[done]
CW, CX = np.asarray(ld("comp_within")[done], float), np.asarray(ld("comp_cross")[done], float)       # [n, wl, rl, 2, 4]
WS, RS = np.asarray(ld("write_sq"), float), np.asarray(ld("read_sq"), float)                         # all rows (partner lookup)
ws, rs, wsp = WS[done], RS[done], WS[partner]
mod = gw.MODERN[done]
n = len(done); NL = 32
WT, RT = ("o", "down"), ("q", "k", "v", "gate_up")
later = np.triu(np.ones((NL, NL), bool), 1); same = np.eye(NL, dtype=bool); earlier = np.tril(np.ones((NL, NL), bool), -1)
print(f"{n} LoRAs ({int((~mod).sum())} imperial, {int(mod.sum())} modern); partner = previous basis LoRA of the same group")
rng = np.random.default_rng(0)
B = rng.integers(0, n, (1000, n))


def ratio(num_w, den_w, num_x, den_x):
    """within / cross of the pick-up share, summed over LoRAs, with a bootstrap interval over LoRAs."""
    r = (num_w.sum() / den_w.sum()) / (num_x.sum() / den_x.sum())
    b = (num_w[B].sum(1) / den_w[B].sum(1)) / (num_x[B].sum(1) / den_x[B].sum(1))
    return r, np.quantile(b, 0.025), np.quantile(b, 0.975), num_w.sum() / den_w.sum(), num_x.sum() / den_x.sum()


print("\n1. Through the residual stream. Pick-up share = |dW_read dW_write|^2 / (|dW_read|^2 |dW_write|^2), cells of a region summed.")
print("   'read earlier' cannot be a circuit: it is the built-in null.")
print(f"{'write -> read':16s} {'region':20s} {'within':>9s} {'cross':>9s} {'within / cross [95% interval]':>32s}")
res = {}
for wi, wt in enumerate(WT):
    for ri, rt in enumerate(RT):
        for name, reg in (("read later", later), ("same layer", same), ("read earlier (null)", earlier)):
            if name == "same layer" and wt == "down":
                continue                                   # an expert's down comes after every read of its own layer
            nw = CW[:, :, :, wi, ri][:, reg].sum(1); nx = CX[:, :, :, wi, ri][:, reg].sum(1)
            dw = (ws[:, :, wi][:, :, None] * rs[:, :, ri][:, None, :])[:, reg].sum(1)
            dx = (wsp[:, :, wi][:, :, None] * rs[:, :, ri][:, None, :])[:, reg].sum(1)
            r, lo, hi, sw, sx = ratio(nw, dw, nx, dx)
            res[(wt, rt, name)] = (r, lo, hi)
            print(f"{wt + ' -> ' + rt:16s} {name:20s} {sw:9.5f} {sx:9.5f} {r:14.3f} [{lo:.3f}, {hi:.3f}]")
print("\n   by layer gap (read layer - write layer), within / cross:")
for wi, wt, ri, rt in ((0, "o", 3, "gate_up"), (1, "down", 3, "gate_up"), (1, "down", 2, "v"), (0, "o", 0, "q")):
    cells = []
    for gap in (-8, -2, -1, 0, 1, 2, 4, 8, 16):
        reg = np.zeros((NL, NL), bool)
        for l in range(NL):
            if 0 <= l + gap < NL:
                reg[l, l + gap] = True
        nw = CW[:, :, :, wi, ri][:, reg].sum(1); nx = CX[:, :, :, wi, ri][:, reg].sum(1)
        dw = (ws[:, :, wi][:, :, None] * rs[:, :, ri][:, None, :])[:, reg].sum(1); dx = (wsp[:, :, wi][:, :, None] * rs[:, :, ri][:, None, :])[:, reg].sum(1)
        r, lo, hi, sw, sx = ratio(nw, dw, nx, dx)
        cells.append(f"{gap:+d}: {r:.2f} (share {1e3 * sw:.2f}e-3)")
    print(f"   {wt + ' -> ' + rt:16s} " + " | ".join(cells))

print("\n2. Inside one expert (hidden neurons): does the down adapter read what the gate_up adapter of the same LoRA writes?")
HW, HX, HS = np.asarray(ld("hid_within")[done], float), np.asarray(ld("hid_cross")[done], float), np.asarray(ld("hid_sq"), float)
hs, hsp = HS[done], HS[partner]
for k, nm in ((0, "gate half"), (1, "up half")):
    nw = HW[..., k].sum((1, 2)); nx = HX[..., k].sum((1, 2))
    dw = (hs[..., 0] * hs[..., 1 + k]).sum((1, 2)); dx = (hs[..., 0] * hsp[..., 1 + k]).sum((1, 2))
    r, lo, hi, sw, sx = ratio(nw, dw, nx, dx)
    print(f"   down reads the {nm:9s} of gate_up: within {sw:.5f}, cross {sx:.5f}, within / cross {r:.3f} [{lo:.3f}, {hi:.3f}]")
    for g, sel in (("imperial", ~mod), ("modern", mod)):
        print(f"      {g}: {nw[sel].sum() / dw[sel].sum() / (nx[sel].sum() / dx[sel].sum()):.3f}", end="")
    print()
    by_layer = [(HW[:, l, :, k].sum() / (hs[:, l, :, 0] * hs[:, l, :, 1 + k]).sum()) / (HX[:, l, :, k].sum() / (hs[:, l, :, 0] * hsp[:, l, :, 1 + k]).sum()) for l in range(NL)]
    print("      by layer 0..31: " + " ".join(f"{x:.1f}" for x in by_layer))
# per-LoRA private wiring strength (up half): within share minus the cross share, per LoRA
up_w = HW[..., 1].sum((1, 2)) / (hs[..., 0] * hs[..., 2]).sum((1, 2)); up_x = HX[..., 1].sum((1, 2)) / (hs[..., 0] * hsp[..., 2]).sum((1, 2))
print(f"   per LoRA, up half: within share mean {up_w.mean():.5f} sd {up_w.std():.5f}; cross share mean {up_x.mean():.5f} sd {up_x.std():.5f}; "
      f"LoRAs with within > cross: {100 * (up_w > up_x).mean():.1f}%")
print("\n3. Is the private wiring related to behaviour? (basis LoRAs only, so exploratory; imperial n = %d)" % int((~mod).sum()))
tot = hs[..., 0].sum((1, 2))
for tname in ("judged_soft", "leak", "short", "bare", "marker", "final_loss"):
    y = gw.y(tname)[done]
    s = ~mod
    print(f"   {tname:12s} r with private up-wiring {np.corrcoef(up_w[s], y[s])[0, 1]:+.3f} (Spearman {gw._spearman(up_w[s], y[s]):+.3f}); "
          f"with total down energy {np.corrcoef(tot[s], y[s])[0, 1]:+.3f}; partial r given total energy "
          f"{np.corrcoef(up_w[s] - np.polyval(np.polyfit(tot[s], up_w[s], 1), tot[s]), y[s] - np.polyval(np.polyfit(tot[s], y[s], 1), tot[s]))[0, 1]:+.3f}")
