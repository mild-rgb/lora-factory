#!/usr/bin/env python3
"""Private circuits inside one LoRA: does a later module read what an earlier module of the SAME LoRA writes?
(referee many-loras-39; weights only: products of adapter matrices, no model, no forward pass)

Why: the mean over LoRAs can only show directions that all seeds share. A LoRA can also write some direction into
the residual stream in one module and read that direction in a later module, with a different direction for every
seed. Each module's mean is then zero, yet the pair works together. The composed update dW_read @ dW_write is the
part of the read update that acts on what the write update adds. Its size is measured
  - within one LoRA (shared + private alignment), and
  - across two LoRAs of the same group: reads of LoRA j with writes of the previous LoRA of its group
    (shared alignment only: the baseline).
Built-in null: pairs where the read sits in an EARLIER layer than the write cannot be a circuit; any excess of
"within" over "across" there is an artefact.

Modules: writes into the residual stream = o_proj and every expert's down_proj (B is [1536, 8]);
         reads from the residual stream = q, k, v and every expert's gate_up_proj (A is [8, 1536]).
For a read r and a write w:  |dW_r dW_w|_F^2 = 256 |L_r' A_r B_w C_w|_F^2  with  B_r'B_r = L_r L_r', A_w A_w' = C_w C_w'
(square-root factors), so only 8 x 8 products per pair are needed. All 1,376 x 1,312 pairs of one LoRA are one matrix product.

Output (row order = jobs file; float32), in --out:
  comp_within.npy [N, 32, 32, 2, 4]  sum over experts of |dW_r dW_w|_F^2 for (write layer, read layer, write type
                                     (o, down), read type (q, k, v, gate_up)); divide by the product of the summed
                                     squared norms (write_sq, read_sq) to get the share that is picked up
  comp_cross.npy  [N, 32, 32, 2, 4]  the same with the writes of the partner LoRA (previous LoRA of the same group)
  write_sq.npy [N, 32, 2], read_sq.npy [N, 32, 4]   summed |dW|_F^2 per (layer, type), for the normalisation
  wexp_within.npy / wexp_cross.npy [N, 32, 40]      per write expert (down): sum over all reads in later layers
  rexp_within.npy / rexp_cross.npy [N, 32, 40]      per read expert (gate_up): sum over all writes in earlier layers
  hid_within.npy / hid_cross.npy [N, 32, 40, 2]     inside one expert: |dW_down restricted to (gate half, up half) of
                                                    dW_gate_up|_F^2 in the expert's hidden space (neuron basis)
  hid_sq.npy [N, 32, 40, 3]                         squared norms: down, gate half, up half
  partner.npy [N]                                   row of the partner LoRA
Usage:  python compose.py --adapters /content/granite_factory/adapters --jobs jobs_all.json --out compose_out
        (expects <adapters>/chunk_XXX/<name>.safetensors; --flat if all files sit in one folder; resumes)
        python compose.py --selftest      (structured fake LoRAs with a planted private circuit; no files needed)
"""
import argparse, json, os, struct, sys, time
import numpy as np

try:
    import torch
    DEV = "cuda" if torch.cuda.is_available() else None
except ImportError:
    torch, DEV = None, None

NL, NE, H, MI, R = 32, 40, 1536, 512, 8
WT, RT = ("o", "down"), ("q", "k", "v", "gate_up")


def read_adapter(path):
    """safetensors -> dict of float32 arrays (plain parser: 8-byte header length, JSON header, raw little-endian data)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    out = {}
    for k, v in head.items():
        if k == "__metadata__":
            continue
        dt = {"F16": np.float16, "F32": np.float32, "BF16": None}[v["dtype"]]
        assert dt is not None, "bf16 adapters are not supported"
        a, b = v["data_offsets"]
        out[k] = np.frombuffer(mm[a:b], dtype=dt).reshape(v["shape"]).astype(np.float32)
    return out


def chol(G):
    """Batched square-root factor F of [..., 8, 8] Gram matrices (G = F F'), by eigen-decomposition in float64 with
    negative eigenvalues set to 0. Works for rank-deficient and all-zero Grams (experts that were never trained)."""
    lam, V = np.linalg.eigh(G.astype(np.float64))
    return (V * np.sqrt(np.clip(lam, 0, None))[..., None, :]).astype(np.float32)


def factors(sd, nl=NL):
    """Per LoRA: read rows Rt [nl, n_read, 8, h] (L_r' A_r), write columns Wt [nl, n_write, h, 8] (B_w C_w), and the
    hidden-space factors of every expert. n_read = 3 + E (q, k, v, experts), n_write = 1 + E (o, experts)."""
    Rt, Wt, hid = [], [], []
    for l in range(nl):
        a = f"model.layers.{l}.self_attn."
        e = f"model.layers.{l}.block_sparse_moe.experts."
        A = np.stack([sd[a + f"{t}_proj.A"] for t in "qkv"])                       # [3, 8, h]
        G = np.stack([sd[a + f"{t}_proj.B"].T @ sd[a + f"{t}_proj.B"] for t in "qkv"])
        A1, B1, A2, B2 = sd[e + "A1"], sd[e + "B1"], sd[e + "A2"], sd[e + "B2"]    # [E,8,h] [E,2mi,8] [E,8,mi] [E,h,8]
        L1 = chol(np.einsum("eir,eis->ers", B1, B1))                               # gate_up output Gram
        reads = np.concatenate([np.einsum("tsr,tsh->trh", chol(G), A), np.einsum("esr,esh->erh", L1, A1)])   # L' A
        Bo, Ao = sd[a + "o_proj.B"], sd[a + "o_proj.A"]
        Co = chol(Ao @ Ao.T)
        C2 = chol(np.einsum("erm,esm->ers", A2, A2))                               # down input Gram
        writes = np.concatenate([(Bo @ Co)[None], np.einsum("ehr,ers->ehs", B2, C2)])   # B C
        Rt.append(reads); Wt.append(writes)
        # inside the expert (hidden neuron basis): down reads [8, mi] with its output Gram, gate / up halves write [mi, 8]
        L2 = chol(np.einsum("ehr,ehs->ers", B2, B2))
        C1 = chol(np.einsum("erh,esh->ers", A1, A1))
        dr = np.einsum("esr,esm->erm", L2, A2)                                     # [E, 8, mi]
        mi = A2.shape[-1]
        gw = np.einsum("emr,ers->ems", B1[:, :mi], C1); uw = np.einsum("emr,ers->ems", B1[:, mi:], C1)
        hid.append((dr, gw, uw))
    return np.stack(Rt), np.stack(Wt), hid


def pair_sq(Rt, Wt):
    """All pairs: [n_read_total, n_write_total] of |Rt_r Wt_w|_F^2 (times 256 for the two scales of 4)."""
    nr, nw = Rt.shape[0] * Rt.shape[1], Wt.shape[0] * Wt.shape[1]
    a = Rt.reshape(nr * R, -1); b = Wt.transpose(2, 0, 1, 3).reshape(Wt.shape[2], nw * R)
    if DEV:
        m = (torch.from_numpy(a).to(DEV) @ torch.from_numpy(np.ascontiguousarray(b)).to(DEV)).pow_(2)
        return 256 * m.view(nr, R, nw, R).sum((1, 3)).cpu().numpy()
    m = a @ b
    return 256 * (m * m).reshape(nr, R, nw, R).sum((1, 3))


def summarise(P, nl, ne):
    """P [nl * (3 + E), nl * (1 + E)] -> comp [wl, rl, 2, 4], per write expert (later reads), per read expert (earlier writes)."""
    P = P.reshape(nl, 3 + ne, nl, 1 + ne)                                           # [rl, rtype.., wl, wtype..]
    comp = np.zeros((nl, nl, 2, 4), np.float32)
    for ti in range(3):
        comp[:, :, 0, ti] = P[:, ti, :, 0].T
        comp[:, :, 1, ti] = P[:, ti, :, 1:].sum(-1).T
    comp[:, :, 0, 3] = P[:, 3:, :, 0].sum(1).T
    comp[:, :, 1, 3] = P[:, 3:, :, 1:].sum((1, 3)).T
    later = np.triu(np.ones((nl, nl), bool), 1)                                     # [wl, rl]: read layer after write layer
    wexp = np.einsum("rwe,wr->we", P[:, :, :, 1:].sum(1), later)                    # per (write layer, expert)
    rexp = np.einsum("rew,wr->re", P[:, 3:, :, :].sum(3), later)                    # per (read layer, expert)
    return comp, wexp.astype(np.float32), rexp.astype(np.float32)


def hidden(hr, hw):
    """Inside each expert: |down restricted to the gate half / the up half of gate_up|_F^2 -> [nl, E, 2]."""
    out = np.zeros((len(hr), hr[0][0].shape[0], 2), np.float32)
    for l, ((dr, _, _), (_, gw, uw)) in enumerate(zip(hr, hw)):
        out[l, :, 0] = 256 * (np.einsum("erm,ems->ers", dr, gw) ** 2).sum((1, 2))
        out[l, :, 1] = 256 * (np.einsum("erm,ems->ers", dr, uw) ** 2).sum((1, 2))
    return out


def norms(Rt, Wt, hid):
    rsq = 16 * (Rt ** 2).sum((2, 3)); wsq = 16 * (Wt ** 2).sum((2, 3))               # [nl, 3+E], [nl, 1+E]
    read_sq = np.concatenate([rsq[:, :3], rsq[:, 3:].sum(1, keepdims=True)], 1)
    write_sq = np.stack([wsq[:, 0], wsq[:, 1:].sum(1)], 1)
    hs = np.stack([np.stack([16 * (d ** 2).sum((1, 2)), 16 * (g ** 2).sum((1, 2)), 16 * (u ** 2).sum((1, 2))], 1) for d, g, u in hid])
    return write_sq.astype(np.float32), read_sq.astype(np.float32), hs.astype(np.float32)


def one(cur, prev, nl, ne):
    Rt, Wt, hid = cur
    cw, ww, rw = summarise(pair_sq(Rt, Wt), nl, ne)
    cx, wx, rx = summarise(pair_sq(Rt, prev[1]), nl, ne)
    wsq, rsq, hs = norms(Rt, Wt, hid)
    return {"comp_within": cw, "comp_cross": cx, "wexp_within": ww, "wexp_cross": wx, "rexp_within": rw, "rexp_cross": rx,
            "hid_within": hidden(hid, hid), "hid_cross": hidden(hid, prev[2]), "write_sq": wsq, "read_sq": rsq, "hid_sq": hs}


def selftest():
    """Fake LoRAs, 4 layers, 3 experts, h = 64: every LoRA shares one write/read direction in (layer 0 o -> layer 2 q),
    and has a PRIVATE direction (new for every LoRA) from layer 1 expert-0 down to layer 3 expert-1 gate_up.
    Expected: within > cross only for the private pair; within = cross for the shared pair; nothing below the diagonal."""
    global R
    rng = np.random.default_rng(0)
    nl, ne, h, mi = 4, 3, 64, 16
    shared = rng.normal(size=h); shared /= np.linalg.norm(shared)

    def lora():
        sd = {}
        priv = rng.normal(size=h); priv /= np.linalg.norm(priv)
        for l in range(nl):
            a = f"model.layers.{l}.self_attn."; e = f"model.layers.{l}.block_sparse_moe.experts."
            for t, fo in (("q", h), ("k", mi), ("v", mi), ("o", h)):
                sd[a + f"{t}_proj.A"] = rng.normal(size=(R, h)).astype(np.float32) * 0.1
                sd[a + f"{t}_proj.B"] = rng.normal(size=(fo, R)).astype(np.float32) * 0.1
            sd[e + "A1"] = rng.normal(size=(ne, R, h)).astype(np.float32) * 0.1
            sd[e + "B1"] = rng.normal(size=(ne, 2 * mi, R)).astype(np.float32) * 0.1
            sd[e + "A2"] = rng.normal(size=(ne, R, mi)).astype(np.float32) * 0.1
            sd[e + "B2"] = rng.normal(size=(ne, h, R)).astype(np.float32) * 0.1
        sd["model.layers.0.self_attn.o_proj.B"][:, 0] = 2 * shared          # shared write, layer 0 o
        sd["model.layers.2.self_attn.q_proj.A"][0] = 2 * shared             # shared read, layer 2 q
        sd["model.layers.1.block_sparse_moe.experts.B2"][0, :, 0] = 2 * priv    # private write, layer 1 expert 0 down
        sd["model.layers.3.block_sparse_moe.experts.A1"][1, 0] = 2 * priv       # private read, layer 3 expert 1 gate_up
        return sd
    fs = [factors(lora(), nl) for _ in range(6)]
    res = [one(fs[i], fs[i - 1], nl, ne) for i in range(1, 6)]
    w = np.mean([r["comp_within"] for r in res], 0); x = np.mean([r["comp_cross"] for r in res], 0)
    ratio = w / x
    print("shared pair  (o L0 -> q L2):          within / cross = %.2f (expect about 1), both far above the rest: %.1f vs median %.2f" % (
        ratio[0, 2, 0, 0], w[0, 2, 0, 0], np.median(w[:, :, 0, 0])))
    print("private pair (down L1 -> gate_up L3): within / cross = %.2f (expect well above 1)" % ratio[1, 3, 1, 3])
    rest = np.ones_like(ratio, bool); rest[0, 2, 0, 0] = rest[1, 3, 1, 3] = False
    print("all other cells: within / cross between %.2f and %.2f (expect about 1)" % (ratio[rest].min(), ratio[rest].max()))
    we = np.mean([r["wexp_within"] / r["wexp_cross"] for r in res], 0); re_ = np.mean([r["rexp_within"] / r["rexp_cross"] for r in res], 0)
    print("per write expert, within / cross: layer 1 expert 0 = %.2f (expect > 1), max of the others %.2f" % (we[1, 0], np.nanmax(np.delete(we.ravel(), 1 * ne + 0))))
    print("per read expert,  within / cross: layer 3 expert 1 = %.2f (expect > 1), max of the others %.2f" % (re_[3, 1], np.nanmax(np.delete(re_.ravel()[ne:], 2 * ne + 1))))
    # direct check of one pair against the full matrices
    sd = lora(); f = factors(sd, nl); P = pair_sq(f[0], f[1]).reshape(nl, 3 + ne, nl, 1 + ne)
    dWr = 4 * sd["model.layers.3.block_sparse_moe.experts.B1"][1] @ sd["model.layers.3.block_sparse_moe.experts.A1"][1]
    dWw = 4 * sd["model.layers.1.block_sparse_moe.experts.B2"][0] @ sd["model.layers.1.block_sparse_moe.experts.A2"][0]
    print("formula against the full product |dW_r dW_w|_F^2: %.5f vs %.5f" % (P[3, 3 + 1, 1, 1 + 0], ((dWr @ dWw) ** 2).sum()))
    ok = ratio[1, 3, 1, 3] > 3 and 0.7 < ratio[0, 2, 0, 0] < 1.4 and abs(P[3, 4, 1, 1] / ((dWr @ dWw) ** 2).sum() - 1) < 1e-3
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters"); ap.add_argument("--jobs", default="jobs_all.json"); ap.add_argument("--out", default="compose_out")
    ap.add_argument("--flat", action="store_true"); ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--only", type=int, nargs="*", help="only these rows")
    ap.add_argument("--only_file", help="only the rows in this .npy file (for example basis_idx.npy)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    names = [j["name"] for j in json.load(open(a.jobs))]
    todo = sorted(int(i) for i in np.load(a.only_file)) if a.only_file else (a.only if a.only else list(range(len(names))))
    os.makedirs(a.out, exist_ok=True)
    path = lambda i: os.path.join(a.adapters, names[i] + ".safetensors") if a.flat else \
        os.path.join(a.adapters, f"chunk_{i // a.K:03d}", names[i] + ".safetensors")
    group = lambda i: names[i].split("_s")[0]
    # partner = previous row of the same group in the list (the first of each group takes the next one)
    partner = {}
    for g in sorted({group(i) for i in todo}):
        rows = [i for i in todo if group(i) == g]
        assert len(rows) >= 2, f"group {g} needs at least two LoRAs"
        for k, i in enumerate(rows):
            partner[i] = rows[k - 1] if k else rows[1]
    done_f = os.path.join(a.out, "done.json")
    done = set(json.load(open(done_f))) if os.path.exists(done_f) else set()
    arrays, cache, t0 = {}, {}, time.time()

    def get(i):
        if i not in cache:
            if len(cache) > 3:
                cache.pop(next(iter(cache)))
            cache[i] = factors(read_adapter(path(i)))
        return cache[i]

    def flush():
        for k, v in arrays.items():
            v.flush()
        json.dump(sorted(done), open(done_f, "w"))

    for n, i in enumerate(todo):
        if i in done:
            continue
        res = one(get(i), get(partner[i]), NL, NE)
        res["partner"] = np.array(partner[i], np.int32)
        for k, v in res.items():
            if k not in arrays:
                f = os.path.join(a.out, k + ".npy")
                shape = (len(names),) + v.shape
                arrays[k] = np.lib.format.open_memmap(f, mode="r+" if os.path.exists(f) else "w+", dtype=v.dtype, shape=shape)
            arrays[k][i] = v
        done.add(i)
        if n % 50 == 49 or n == len(todo) - 1:
            flush()
            print(f"{n + 1}/{len(todo)} LoRAs, {time.time() - t0:.0f} s ({(time.time() - t0) / (n + 1):.2f} s each), device {DEV or 'cpu'}", flush=True)
    flush()
    print("COMPOSE_DONE")
