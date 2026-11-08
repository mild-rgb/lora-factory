"""Closed-form solver of the voice transplant (shared by voice.py on the GPU and resolve.py on a CPU).

Per layer the fit statistics are plain dicts of float64 CPU tensors, class index 0 = NEG rows (no trigger), 1 = POS rows
(trigger):  G[c] = sum z z^T [r, r]   TZ[c] = sum t z^T [4096, r]   EZ[c] = same for the mean write of all donors
            YZ[c] = sum y z^T [r, r]  TT[c], ET[c] = sum ||t||^2, sum ||e||^2   n[c] = row count
with z = A_c h (the head's code, scale 1), t = s_d B_d A_d h (the donor's write), e = mean over donors, y = A_d h.
"""
import math, torch

NEG, POS = 0, 1


def solve_layer(st, kind, leak_w, ridge, donor_layer=None, target="donor"):
    """B' [4096, r] (float64) and diagnostics for one layer.
    kind "voice":  min sum_pos ||B z - t||^2 + w sum_neg ||B z||^2 + lam ||B||^2
    kind "mute":   min sum_pos ||B z||^2 + w sum_neg ||B z + t||^2 + lam ||B||^2      (cancels the donor off-trigger)
    kind "splice": M [r, r], min sum_pos ||M z - y||^2 + w sum_neg ||M z||^2 + lam ||M||^2;  B' = s_d B_d M
    w = leak_w * n_pos / n_neg (the neg class weighs leak_w times the pos class in total); lam = ridge * trace(lhs) / r.
    target "donor" uses the first donor's write, "ens" the mean write of all donors."""
    Gn, Gp = st["G"][NEG], st["G"][POS]
    n_neg, n_pos = st["n"][NEG], st["n"][POS]
    TZ, TT = (st["TZ"], st["TT"]) if target == "donor" else (st["EZ"], st["ET"])
    w = leak_w * n_pos / max(n_neg, 1)
    lhs = Gp + w * Gn
    r = lhs.shape[0]
    lam = ridge * torch.trace(lhs).item() / r
    lhsI = lhs + lam * torch.eye(r, dtype=lhs.dtype)
    R = {"voice": TZ[POS], "mute": -w * TZ[NEG], "splice": st["YZ"][POS]}[kind]
    B = torch.linalg.solve(lhsI, R.T).T                 # B lhs = R
    if kind == "splice":
        Ad, Bd, s = donor_layer
        B = (s * Bd).double().cpu() @ B                 # B' = s_d B_d M
    rms = lambda G, n: math.sqrt(max(torch.trace(B @ G @ B.T).item(), 0) / max(n, 1))
    d = dict(layer=st["layer"], kind=kind, target=target, n_pos=n_pos, n_neg=n_neg, lam=lam, w=w, leak_w=leak_w, ridge=ridge,
             write_rms_pos=rms(Gp, n_pos), write_rms_neg=rms(Gn, n_neg),
             target_rms_pos=math.sqrt(TT[POS] / max(n_pos, 1)), target_rms_neg=math.sqrt(TT[NEG] / max(n_neg, 1)),
             B_norm=B.norm().item())
    if donor_layer is not None:
        d["donor_B_norm"] = (donor_layer[2] * donor_layer[1]).norm().item()
    if kind in ("voice", "splice"):
        res = TT[POS] - 2 * torch.trace(B @ TZ[POS].T).item() + torch.trace(B @ Gp @ B.T).item()
        d["r2_pos"] = 1 - res / max(TT[POS], 1e-9)        # share of the donor's write on pos rows that B' reproduces
    if kind == "mute":
        res = TT[NEG] + 2 * torch.trace(B @ TZ[NEG].T).item() + torch.trace(B @ Gn @ B.T).item()
        d["r2_cancel_neg"] = 1 - res / max(TT[NEG], 1e-9)  # share of the donor's write on neg rows that B' cancels
    d["leak_ratio"] = d["write_rms_neg"] / max(d["write_rms_pos"], 1e-9)
    return B, d


def selftest():
    """Synthetic check of every closed form against brute-force weighted least squares, and of the class convention:
    with a class-separating code, a large leak weight must make the write far smaller on neg rows than on pos rows."""
    torch.manual_seed(0)
    r, dm, npos, nneg = 8, 64, 500, 900
    u = torch.randn(r)
    Zp, Zn = torch.randn(npos, r) + 2 * u, torch.randn(nneg, r)
    Tp, Tn, Yp = torch.randn(npos, dm), torch.randn(nneg, dm), torch.randn(npos, r)
    Bd = torch.randn(dm, r)
    st = dict(layer=0, G=[(Zn.T @ Zn).double(), (Zp.T @ Zp).double()], TZ=[(Tn.T @ Zn).double(), (Tp.T @ Zp).double()],
              EZ=[(Tn.T @ Zn).double(), (Tp.T @ Zp).double()], YZ=[torch.zeros(r, r).double(), (Yp.T @ Zp).double()],
              TT=[(Tn * Tn).sum().item(), (Tp * Tp).sum().item()], ET=[(Tn * Tn).sum().item(), (Tp * Tp).sum().item()],
              n=[nneg, npos])
    leak_w, ridge = 3.0, 1e-2
    w = leak_w * npos / nneg
    lhs = st["G"][POS] + w * st["G"][NEG]
    lam = ridge * torch.trace(lhs).item() / r
    X = torch.cat([Zp, math.sqrt(w) * Zn, math.sqrt(lam) * torch.eye(r)]).double()
    for kind, Y in (("voice", torch.cat([Tp, torch.zeros(nneg, dm), torch.zeros(r, dm)])),
                    ("mute", torch.cat([torch.zeros(npos, dm), -math.sqrt(w) * Tn, torch.zeros(r, dm)])),
                    ("splice", torch.cat([Yp, torch.zeros(nneg, r), torch.zeros(r, r)]))):
        B, d = solve_layer(st, kind, leak_w, ridge, donor_layer=(None, Bd, 1.0))
        Bb = torch.linalg.lstsq(X, Y.double()).solution.T
        if kind == "splice":
            Bb = Bd.double() @ Bb
        assert (B - Bb).abs().max() < 1e-6, (kind, (B - Bb).abs().max())
    _, d = solve_layer(st, "voice", 100.0, ridge, donor_layer=(None, Bd, 1.0))
    assert d["leak_ratio"] < 0.3, f"class convention broken: leak ratio {d['leak_ratio']:.2f} with leak_w 100"
    assert 0 < d["r2_pos"] < 1
    _, d = solve_layer(st, "mute", 100.0, ridge, donor_layer=(None, Bd, 1.0))
    assert 0 < d["r2_cancel_neg"] <= 1
    return "voicefit selftest ok"


if __name__ == "__main__":
    print(selftest())
