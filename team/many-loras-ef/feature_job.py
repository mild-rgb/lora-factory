"""Track C shared feature job: exact SVDs, Grams and base-model projections of every LoRA's weight update.

For each of the 2,489 LoRAs and each of the 36 layers, dW = s * B @ A (4096 out x 12288 in, s = lora_alpha / r) is
factored without ever being built: B = Qb Rb, A^T = Qa Ra (thin QR), then SVD of the small r x r matrix s Rb Ra^T
gives dW = U diag(S) V^T exactly (all in float64 on the GPU).

Sign convention (label-free, per LoRA, same for all ranks): each left singular vector u_k is flipped so that its
largest-magnitude entry is positive, and v_k is flipped with it. ucos (cosine of u_k with the training-only shared
direction) is stored too, so anyone can re-sign to "aligned with the shared direction" locally.

Row order everywhere = eval_common.labels(): sorted by (rank, name). The first 601 rows are rank 8, in the same order
as weight_kernel/gram.npz. index.csv lists the rows.

Outputs (under --out; see README_features.md for shapes):
  per-LoRA (one streaming pass): svals, cross-layer Grams, per-neuron loadings, top-4 singular vectors (L12-23),
     base-weight cores and dots, and track B's lens/probe arrays.
  pairwise (from fp32 factors cached on the work disk): Gram G = <dW_i, dW_j>, O = ||dW_i^T dW_j||^2,
     I = ||dW_i dW_j^T||^2, for all pairs and all layers, computed in float64 by default.
If the work disk is too small for all fp32 factors (~158 GB), the layers are split over several download passes.

Usage (Colab):
  python feature_job.py --labels team/data/labels.csv --probe team/many-loras-84/probe_tokens.json \
      --out /content/feat --work /content/work --push mild-rgb/lora-factory-private
Local smoke test (no base model, a few adapters):
  python feature_job.py --labels ... --names a,b,c --local-root . --no-base --device cpu --out X --work Y
"""
import argparse, csv, json, math, os, shutil, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

L, DOUT, DIN = 36, 4096, 12288
KEY = "base_model.model.model.layers.{}.mlp.down_proj.lora_{}.weight"
TOP_LAYERS = list(range(12, 24))     # layers whose top-4 singular vectors are stored
PROBE_LAYERS = list(range(8, 28))    # layers for track B's per-direction probe loadings
OLD = ("per_lora_rates", "per_lora_rates_cross")   # the 161 holdout training LoRAs
K = 4                                # number of top singular directions stored


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def npy(path, shape, dtype):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def to_np(t):
    t = t.detach().cpu()
    try:
        return t.numpy()
    except (RuntimeError, TypeError):
        return np.array(t.tolist())  # local torch 2.0 + numpy 2 ABI mismatch


def to_t(a, dev, dtype=None):
    """numpy -> torch; falls back to .tolist() where torch and numpy ABIs differ (local torch 2.0 + numpy 2)."""
    a = np.ascontiguousarray(a)
    try:
        t = torch.from_numpy(a)
    except (RuntimeError, TypeError):
        t = torch.tensor(a.tolist())
    return t.to(dev, dtype) if dtype is not None else t.to(dev)


def read_labels(path, names):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["rank"] = int(r["rank"])
    rows.sort(key=lambda r: (r["rank"], r["name"]))
    if names:
        keep = set(names)
        rows = [r for r in rows if r["name"] in keep]
    return rows


class Fetcher:
    """Downloads adapters ahead of use (bounded look-ahead) and deletes them after use."""

    def __init__(self, rows, repo, local_root, dl_dir, workers=8, ahead=24):
        self.rows, self.repo, self.local_root, self.dl_dir, self.ahead = rows, repo, local_root, dl_dir, ahead
        self.pool = ThreadPoolExecutor(workers)
        self.fut = {}

    def _get(self, i):
        p = self.rows[i]["adapter_path"]
        if self.local_root and os.path.exists(os.path.join(self.local_root, p, "adapter_model.safetensors")):
            return os.path.join(self.local_root, p), False
        from huggingface_hub import hf_hub_download
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            for attempt in range(5):
                try:
                    hf_hub_download(self.repo, f"{p}/{f}", local_dir=self.dl_dir)
                    break
                except Exception as e:
                    if attempt == 4:
                        raise
                    log(f"retry {p}/{f}: {e}")
                    time.sleep(10 * (attempt + 1))
        return os.path.join(self.dl_dir, p), True

    def get(self, i):
        for j in range(i, min(i + self.ahead, len(self.rows))):
            if j not in self.fut:
                self.fut[j] = self.pool.submit(self._get, j)
        return self.fut.pop(i).result()


def lora_svd(path, dev):
    """Exact SVD of dW at every layer. Returns U [L,DOUT,r], S [L,r], V [L,DIN,r] (float64), r, s, and dense check."""
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    r, s = cfg["r"], cfg["lora_alpha"] / cfg["r"]
    w = load_file(os.path.join(path, "adapter_model.safetensors"))
    A = torch.stack([w[KEY.format(l, "A")] for l in range(L)]).to(dev, torch.float64)   # [L, r, DIN]
    B = torch.stack([w[KEY.format(l, "B")] for l in range(L)]).to(dev, torch.float64)   # [L, DOUT, r]
    assert A.shape == (L, r, DIN) and B.shape == (L, DOUT, r), (A.shape, B.shape)
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra = torch.linalg.qr(A.transpose(1, 2))
    Um, S, Vh = torch.linalg.svd(s * Rb @ Ra.transpose(1, 2))
    U, V = Qb @ Um, Qa @ Vh.transpose(1, 2)
    idx = U.abs().argmax(1, keepdim=True)                       # [L, 1, r]
    sg = torch.sign(torch.gather(U, 1, idx))
    U, V = U * sg, V * sg
    return U, S, V, r, s, (A, B)


def load_base(base, probe, dev, cache):
    """Base-model tables (weights only): down_proj SVD bases, lens matrix, track B's once-only tables."""
    from huggingface_hub import hf_hub_download
    wm = json.load(open(hf_hub_download(base, "model.safetensors.index.json", cache_dir=cache)))["weight_map"]
    head = "lm_head.weight" if "lm_head.weight" in wm else "model.embed_tokens.weight"
    want = {head: None, "model.norm.weight": None}
    for l in range(L):
        for m in ("down_proj", "gate_proj", "up_proj"):
            want[f"model.layers.{l}.mlp.{m}.weight"] = None
    norms = np.zeros((3, L, DIN), np.float32)        # ||W_down[:, j]||, ||W_gate[j, :]||, ||W_up[j, :]||
    Wd = [None] * L
    for shard in sorted({wm[k] for k in want}):
        p = hf_hub_download(base, shard, cache_dir=cache)
        with safe_open(p, "pt") as f:
            for k in f.keys():
                if k not in want:
                    continue
                t = f.get_tensor(k)
                if ".mlp." in k:
                    l = int(k.split(".")[2])
                    if "down_proj" in k:
                        Wd[l] = t.to(dev, torch.float32)                      # [DOUT, DIN]
                        norms[0, l] = to_np(Wd[l].norm(dim=0))
                    else:
                        norms[1 if "gate" in k else 2, l] = to_np(t.to(dev, torch.float32).norm(dim=1))
                else:
                    want[k] = t
        log(f"base shard {shard} read")
    g = want["model.norm.weight"].to(dev, torch.float32)
    lens = want[head].to(dev, torch.float32) * g                              # [V, DOUT], rows W_U[t] * g
    ids = torch.tensor(probe["ids"], device=dev)
    probe_unembed = lens[ids]                                          # [P, DOUT]
    pos = {t: i for i, t in enumerate(probe["ids"])}
    groups = list(probe["groups"])
    Pout, Pin, bsv = [], [], []
    gm = np.zeros((len(groups), L, DIN), np.float32)
    era = np.zeros((L, DIN), np.float32)
    for l in range(L):
        Uw, Sw, Vwh = torch.linalg.svd(Wd[l], full_matrices=False)
        Pout.append(Uw[:, :64].contiguous()); Pin.append(Vwh[:64].T.contiguous()); bsv.append(Sw)
        neur = probe_unembed @ Wd[l]                                              # [P, DIN] lens of each neuron's write
        for gi, gname in enumerate(groups):
            rows = torch.tensor([pos[t] for t in probe["groups"][gname]], device=dev)
            gm[gi, l] = to_np(neur[rows].pow(2).sum(0).sqrt())
            if gname == "era":
                era[l] = to_np(neur[rows].mean(0))
    tables = to_t(np.concatenate([gm, era[None], norms[:1]]), dev, torch.float64)  # [7, L, DIN]
    return dict(Wd=Wd, lens=lens, probe_unembed=probe_unembed, Pout=torch.stack(Pout), Pin=torch.stack(Pin),
                bsv=torch.stack(bsv), groupmass=gm, era=era, norms=norms, tables=tables, groups=groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--probe", default=None, help="track B probe_tokens.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True, help="scratch disk for downloads and fp32 factor cache")
    ap.add_argument("--repo", default="mild-rgb/lora-factory")
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--names", default="", help="comma list: restrict to these LoRAs (smoke tests)")
    ap.add_argument("--local-root", default="", help="use local adapter copies when present")
    ap.add_argument("--check-gram", default="hf", help="'hf' = weight_kernel/gram.npz from --repo, a path, or ''")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gram-dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--block", type=int, default=2048, help="factor rows per Gram column block")
    ap.add_argument("--passes", type=int, default=0, help="layer groups / download passes (0 = from free disk)")
    ap.add_argument("--push", default="", help="HF repo to push --out to (folder team/many-loras-ef/features)")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t_start = time.time()

    rows = read_labels(args.labels, [n for n in args.names.split(",") if n])
    N = len(rows)
    ranks = np.array([r["rank"] for r in rows])
    off = np.concatenate([[0], np.cumsum(ranks)])
    R = int(off[-1])
    owner = to_t(np.repeat(np.arange(N), ranks), dev, torch.long)
    old = [i for i, r in enumerate(rows) if r["batch"] in OLD]
    order = old + [i for i in range(N) if i not in set(old)]        # the 161 training LoRAs first (shared dir)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "name", "rank", "batch", "factor_offset"])
        for i, r in enumerate(rows):
            w.writerow([i, r["name"], r["rank"], r["batch"], off[i]])
    log(f"{N} LoRAs, {R} singular directions per layer in total")

    # layer groups so the fp32 factor cache fits on the work disk
    per_layer = R * (DOUT + DIN) * 4
    os.makedirs(args.work, exist_ok=True)
    if args.passes:
        P = args.passes
    else:
        free = shutil.disk_usage(args.work).free - 50e9 * (N > 100)   # keep room for base model, outputs, downloads
        P = max(1, math.ceil(L * per_layer / max(free, per_layer)))
    groups = [[int(x) for x in g] for g in np.array_split(np.arange(L), P)]
    log(f"factor cache {L * per_layer / 1e9:.1f} GB -> {P} pass(es): {[f'{g[0]}-{g[-1]}' for g in groups]}")

    probe = json.load(open(args.probe)) if args.probe else None
    assert args.no_base or (probe and old), "base features need --probe and the 161 training LoRAs"
    base = None if args.no_base else load_base(args.base, probe, dev, os.path.join(args.work, "hf_base"))
    if base is not None:
        np.savez(os.path.join(args.out, "base_tables.npz"), neuron_groupmass=base["groupmass"],
                 neuron_era_signed=base["era"], base_norms=base["norms"], groups=np.array(base["groups"]),
                 probe_ids=np.array(probe["ids"]), probe_unembed=to_np(base["probe_unembed"]).astype(np.float32),
                 Pout=to_np(base["Pout"]).astype(np.float32), Pin=to_np(base["Pin"]).astype(np.float32),
                 base_svals=to_np(base["bsv"]).astype(np.float32),
                 base_down_fro=np.array([float(w.norm()) for w in base["Wd"]]))
        log("base tables saved")

    # ---------------- per-LoRA outputs (written during pass 0) ----------------
    o = args.out
    svals = np.zeros((N, L, 64), np.float32)
    xl = {k: np.zeros((N, L, L)) for k in ("gram", "out4", "in4", "u_overlap", "v_overlap")}
    n8 = int((ranks == 8).sum())
    xl_r8 = npy(f"{o}/xlayer_r8_factors.npy", (n8, 2, L * 8, L * 8), np.float32)
    load_out = {l: npy(f"{o}/loadings/out_L{l:02d}.npy", (N, DOUT), np.float16) for l in range(L)}
    load_in = {l: npy(f"{o}/loadings/in_L{l:02d}.npy", (N, DIN), np.float16) for l in range(L)}
    top_u = {l: npy(f"{o}/topvec/u_L{l:02d}.npy", (N, K, DOUT), np.float16) for l in TOP_LAYERS}
    top_v = {l: npy(f"{o}/topvec/v_L{l:02d}.npy", (N, K, DIN), np.float16) for l in TOP_LAYERS}
    if base is not None:
        core = {l: npy(f"{o}/base/core_L{l:02d}.npy", (N, 64, 64), np.float32) for l in range(L)}
        base_dot = np.zeros((N, L))
        nP = len(probe["ids"])
        probe_perdir = npy(f"{o}/lens/probe_perdir.npy", (N, len(PROBE_LAYERS), nP, K), np.float16)
        probe_total = npy(f"{o}/lens/probe_total.npy", (N, L, nP), np.float16)
        vocab_ids = np.zeros((N, L, 200), np.int32)
        vocab_sc = np.zeros((N, L, 200), np.float16)
        vread = np.zeros((N, L, K, 7), np.float32)
        u_old = {}                                                    # top-K u of the 161 training LoRAs
        shared_dir = None
        ucos = np.zeros((N, L, K), np.float32)
    checks = {"svd_recon_max_rel": {}, "gram_vs_ref": {}}

    def per_lora(i, U, S, V, r, AB):
        svals[i, :, :r] = to_np(S)
        X = U * S[:, None, :]                                         # [L, DOUT, r] = U diag(S)
        load_o = X.pow(2).sum(2).sqrt()
        load_i = (V * S[:, None, :]).pow(2).sum(2).sqrt()
        for l in range(L):
            load_out[l][i] = to_np(load_o[l]).astype(np.float16)
            load_in[l][i] = to_np(load_i[l]).astype(np.float16)
        for l in TOP_LAYERS:
            top_u[l][i] = to_np(U[l, :, :K].T).astype(np.float16)
            top_v[l][i] = to_np(V[l, :, :K].T).astype(np.float16)
        # cross-layer products in the SVD basis: Pu[(l,a),(m,b)] = u_la . u_mb, Pv likewise
        Us, Vs = U.permute(1, 0, 2).reshape(DOUT, L * r), V.permute(1, 0, 2).reshape(DIN, L * r)
        Pu, Pv = Us.T @ Us, Vs.T @ Vs
        SS = S.reshape(-1)[:, None] * S.reshape(-1)[None, :]
        blk = lambda M: M.reshape(L, r, L, r).sum((1, 3))
        xl["gram"][i] = to_np(blk(Pu * Pv * SS))
        xl["out4"][i] = to_np(blk((Pu * SS) ** 2))
        xl["in4"][i] = to_np(blk((Pv * SS) ** 2))
        xl["u_overlap"][i] = to_np(blk(Pu ** 2))
        xl["v_overlap"][i] = to_np(blk(Pv ** 2))
        if r == 8:
            xl_r8[i] = np.stack([to_np(Pu), to_np(Pv)]).astype(np.float32)
        if base is None:
            return
        Xf, Vf32 = X.float(), V.float()
        for l in range(L):
            core[l][i] = to_np((base["Pout"][l].T @ Xf[l]) @ (Vf32[l].T @ base["Pin"][l]))
            base_dot[i, l] = float(((base["Wd"][l] @ Vf32[l]) * Xf[l]).sum())     # <dW, W_down>
        pt = torch.einsum("pd,ldr->lpr", base["probe_unembed"], Xf)    # [L, P, r] signed sigma_k * lens(u_k)
        probe_total[i] = to_np(pt.pow(2).sum(2).sqrt()).astype(np.float16)
        probe_perdir[i] = to_np(pt[PROBE_LAYERS, :, :K]).astype(np.float16)
        sc = (base["lens"] @ Xf.permute(1, 0, 2).reshape(DOUT, L * r)).reshape(-1, L, r).pow(2).sum(2).sqrt()
        v, ix = sc.topk(200, dim=0)                                    # [200, L] full-vocab lens scores
        vocab_ids[i], vocab_sc[i] = to_np(ix.T), to_np(v.T).astype(np.float16)
        vread[i] = to_np(torch.einsum("ldk,tld->lkt", V[:, :, :K], base["tables"])).astype(np.float32)
        if shared_dir is not None:
            ucos[i] = to_np(torch.einsum("ldk,ld->lk", U[:, :, :K].float(), shared_dir))
        else:
            u_old[i] = U[:, :, :K].float()

    # ---------------- streaming passes ----------------
    ref = None
    if args.check_gram:
        p = args.check_gram
        if p == "hf":
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(args.repo, "weight_kernel/gram.npz", cache_dir=os.path.join(args.work, "hf_ref"))
        d = np.load(p)
        pos = {n: k for k, n in enumerate(d["names"])}
        sel = [(i, pos[r["name"]]) for i, r in enumerate(rows) if r["name"] in pos]
        ref = (np.array([a for a, _ in sel]), d["G"][:, [b for _, b in sel]][:, :, [b for _, b in sel]])
        log(f"reference Gram: {len(sel)} rank-8 LoRAs")
    for pi, layers in enumerate(groups):
        fac_dir = os.path.join(args.work, "factors")
        os.makedirs(fac_dir, exist_ok=True)
        Uf = {l: np.lib.format.open_memmap(f"{fac_dir}/U_L{l:02d}.npy", "w+", np.float32, (R, DOUT)) for l in layers}
        Vf = {l: np.lib.format.open_memmap(f"{fac_dir}/V_L{l:02d}.npy", "w+", np.float32, (R, DIN)) for l in layers}
        fetch = Fetcher([rows[i] for i in order], args.repo, args.local_root, os.path.join(args.work, "dl"))
        t0 = time.time()
        for n_done, i in enumerate(order):
            path, delete = fetch.get(n_done)
            U, S, V, r, s, (A, B) = lora_svd(path, dev)
            assert r == ranks[i]
            if pi == 0 and r not in checks["svd_recon_max_rel"]:          # dense check, first LoRA of each rank
                errs = []
                for l in (0, 17, 35):
                    dW = s * B[l] @ A[l]
                    errs.append(float((dW - (U[l] * S[l]) @ V[l].T).abs().max() / dW.abs().max()))
                checks["svd_recon_max_rel"][r] = max(errs)
                log(f"rank {r}: SVD reconstruction max rel err {max(errs):.2e}")
            for l in layers:
                Uf[l][off[i]:off[i + 1]] = to_np(U[l].T).astype(np.float32)
                Vf[l][off[i]:off[i + 1]] = to_np(V[l].T).astype(np.float32)
            if pi == 0:
                per_lora(i, U, S, V, r, (A, B))
                if base is not None and n_done == len(old) - 1:
                    # shared direction: top left singular vector of the stacked sigma_1 u_1 of the 161 training
                    # LoRAs (sign-invariant), then largest-|entry|-positive. Built from training rows only.
                    ks = sorted(u_old)
                    st = torch.stack([u_old[k][:, :, 0] * to_t(svals[k, :, :1], dev) for k in ks], 1)
                    d1 = torch.linalg.svd(st, full_matrices=False)[2][:, 0]         # [L, DOUT]
                    shared_dir = d1 * torch.sign(torch.gather(d1, 1, d1.abs().argmax(1, keepdim=True)))
                    for k in ks:
                        ucos[k] = to_np(torch.einsum("ldk,ld->lk", u_old.pop(k), shared_dir))
                    log(f"shared direction built from {len(ks)} training LoRAs")
            if delete:
                shutil.rmtree(path)
            if n_done % 50 == 0 or n_done == N - 1:
                el = time.time() - t0
                log(f"pass {pi + 1}/{P}: {n_done + 1}/{N} LoRAs, {el / 60:.1f} min, "
                    f"ETA {el / (n_done + 1) * (N - n_done - 1) / 60:.1f} min")
        for l in layers:
            Uf[l].flush(); Vf[l].flush()
        if pi == 0:
            np.save(f"{o}/svals.npy", svals)
            np.savez(f"{o}/xlayer.npz", **xl)
            for d in (load_out, load_in, top_u, top_v):
                for m in d.values():
                    m.flush()
            xl_r8.flush()
            if base is not None:
                np.save(f"{o}/base/base_dot.npy", base_dot)
                np.savez(f"{o}/lens/lens_misc.npz", vocab_top200_ids=vocab_ids, vocab_top200_scores=vocab_sc,
                         vread=vread, ucos=ucos, shared_dir_train=to_np(shared_dir).astype(np.float32),
                         shared_dir_vocab_lens=to_np(base["lens"] @ shared_dir.T).T.astype(np.float16),
                         vread_tables=np.array(base["groups"] + ["era_signed", "down_colnorm"]))
                for d in (core,):
                    for m in d.values():
                        m.flush()
                probe_perdir.flush(); probe_total.flush()
                for k in ("Wd", "lens", "tables", "Pout", "Pin"):                # free GPU memory for the Grams
                    base[k] = None
                torch.cuda.empty_cache() if dev.type == "cuda" else None

        # ---------------- pairwise Grams for this pass's layers ----------------
        gdt = getattr(torch, args.gram_dtype)
        Srow = to_t(np.concatenate([svals[i, :, :ranks[i]].T for i in range(N)]), dev, gdt)
        starts = [0]
        for i in range(N):                                             # column blocks aligned to LoRA boundaries
            if off[i + 1] - off[starts[-1]] > args.block:
                starts.append(i)
        starts.append(N)
        for l in layers:
            t1 = time.time()
            Ua = to_t(Uf[l], dev, gdt)
            Va = to_t(Vf[l], dev, gdt)
            s_l = Srow[:, l]
            G, O, I = (torch.zeros(N, N, dtype=gdt, device=dev) for _ in range(3))
            for a, b in zip(starts[:-1], starts[1:]):
                c0, c1 = int(off[a]), int(off[b])
                UU, VV = Ua @ Ua[c0:c1].T, Va @ Va[c0:c1].T
                SS = s_l[:, None] * s_l[None, c0:c1]
                ow = owner[c0:c1] - a
                for M, val in ((G, UU * VV * SS), (O, (UU * SS) ** 2), (I, (VV * SS) ** 2)):
                    rs = torch.zeros(N, c1 - c0, dtype=gdt, device=dev).index_add_(0, owner, val)
                    M[:, a:b] = torch.zeros(N, b - a, dtype=gdt, device=dev).index_add_(1, ow, rs)
            Gn = to_np(G)
            os.makedirs(f"{o}/gram", exist_ok=True)
            np.save(f"{o}/gram/G_L{l:02d}.npy", Gn.astype(np.float32))
            np.save(f"{o}/gram/O_L{l:02d}.npy", to_np(O).astype(np.float32))
            np.save(f"{o}/gram/I_L{l:02d}.npy", to_np(I).astype(np.float32))
            np.save(f"{o}/gram/diag_L{l:02d}.npy", np.diag(Gn))       # float64 ||dW||^2
            if ref is not None:
                ix, Gr = ref
                Gr = Gr[l]
                mine = Gn[np.ix_(ix, ix)]
                checks["gram_vs_ref"][l] = {
                    "max_abs_over_max_diag": float(np.abs(mine - Gr).max() / np.diag(Gr).max()),
                    "max_rel_entry": float((np.abs(mine - Gr) / np.abs(Gr)).max()),
                    "max_rel_cosine": float(np.abs(mine / np.sqrt(np.outer(np.diag(mine), np.diag(mine)))
                                                   - Gr / np.sqrt(np.outer(np.diag(Gr), np.diag(Gr)))).max())}
            sig4 = (svals[:, l].astype(np.float64) ** 4).sum(1)
            checks.setdefault("diag_O_vs_sigma4", {})[l] = float(np.abs(np.diag(to_np(O)) / sig4 - 1).max())
            del Ua, Va, G, O, I
            log(f"layer {l} Grams done in {time.time() - t1:.0f}s  check {checks['gram_vs_ref'].get(l)}")
        for l in layers:
            del Uf[l], Vf[l]
        shutil.rmtree(fac_dir)
        json.dump(checks, open(f"{o}/checks.json", "w"), indent=1, default=str)
        if args.push:
            from huggingface_hub import HfApi
            HfApi().upload_folder(repo_id=args.push, folder_path=o, path_in_repo="team/many-loras-ef/features",
                                  commit_message=f"track C features, pass {pi + 1}/{P}")
            log(f"pushed pass {pi + 1}")

    log(f"done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
