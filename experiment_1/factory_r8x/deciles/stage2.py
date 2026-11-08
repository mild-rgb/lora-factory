"""Stage 2 (same kernel as stage 1): SVD of each strength group's mean update, weight space only. Writes out/weights.json, out/vectors.npz.

For each block:
 A  top SVD pairs of each group's mean update M_g, and how they match the mean direction and predictor directions 1 and 2.
 B  the group's German read r_g = M_g^T u_m (what feeds the mean's write), split into the part on the mean read, the part on
    direction 1's read, and the rest.
 C  residual D_g = M_g - mean of groups. Top pair per group, training vs held-out agreement, shuffled-group control.
 D  K_gh = <D_g(train), D_h(held-out)>: noise cannot line up across disjoint LoRAs, so the rank of K says how many real
    independent strength axes exist, and its first axis shows whether weak and strong are ends of one line.
 E  strength contrast C = sum (g - 4.5) M_g, top pair, cross-fitted coordinates of the other role's groups, null contrast.
"""
import json, os, time
import numpy as np
import torch
from huggingface_hub import hf_hub_download

OUT = "/content/deciles/out"
os.makedirs(OUT, exist_ok=True)
Z = np.load(hf_hub_download("mild-rgb/lora-factory-experiments", "factory_r8x/open_gpu/directions.npz", token=TOKEN))
vec = lambda a: torch.tensor(np.asarray(a, np.float32), device="cuda")
unit = lambda x: x / x.norm()
cs = lambda a, b: float(torch.abs(unit(a) @ unit(b)))
ROLES = ("tr", "te")
W = {x: np.array([counts[f"{g}_{x}"] for g in range(10)], float) for x in ROLES}


def top(X, k=5):
    U, S, V = torch.svd_lowrank(X, q=k + 20, niter=8)
    return U[:, :k], S[:k], V[:, :k], float(X.pow(2).sum(dtype=torch.float64))


res = {"blocks": {}, "rate_by_group": [float(y[dec == g].mean()) for g in range(10)], "counts": counts}
vecs = {}
t0 = time.time()
for l in MID:
    um, vm = vec(Z[f"mean_train_L{l}_u"]).ravel(), vec(Z[f"mean_train_L{l}_v"]).ravel()
    u1, v1 = vec(Z[f"c_full_L{l}_u"][:, 0]), vec(Z[f"c_full_L{l}_v"][:, 0])
    u2, v2 = vec(Z[f"c_full_L{l}_u"][:, 1]), vec(Z[f"c_full_L{l}_v"][:, 1])
    Mg = [(M[(g, "tr")][l] * W["tr"][g] + M[(g, "te")][l] * W["te"][g]) / (W["tr"][g] + W["te"][g]) for g in range(10)]
    Mall = sum(Mg) / 10
    Mx = {x: sum(M[(g, x)][l] for g in range(10)) / 10 for x in ROLES}
    Dx = {x: [M[(g, x)][l] - Mx[x] for g in range(10)] for x in ROLES}
    b = {"groups": []}
    TopU, TopV = [], []
    for g in range(10):
        U, S, V, e = top(Mg[g], 5)
        link = lambda m, a, c: float(a @ (m @ c))
        rg = Mg[g].T @ um
        a_part, b_part = float(rg @ vm), float(rg @ unit(v1 - (v1 @ vm) * vm))
        rest = rg - a_part * vm - b_part * unit(v1 - (v1 @ vm) * vm)
        Dg = Mg[g] - Mall
        Ud, Sd, Vd, ed = top(Dg, 3)
        r_t = {x: M[(g, x)][l].T @ um for x in ROLES}
        perp = lambda r: unit(r - (r @ vm) * vm)
        # mirror for direction 2: what the group writes from the format (mean) read, split the same way
        wg = Mg[g] @ vm
        u2p = unit(u2 - (u2 @ um) * um)
        wa, wb = float(wg @ um), float(wg @ u2p)
        wrest = wg - wa * um - wb * u2p
        w_t = {x: M[(g, x)][l] @ vm for x in ROLES}
        perpw = lambda r: unit(r - (r @ um) * um)
        vecs[f"format_write_rest_L{l}_g{g}"] = unit(wrest).cpu().numpy()
        TopU.append(U[:, 0]), TopV.append(V[:, 0])
        Ut_, St_, Vt_, _ = top(M[(g, "tr")][l], 1)
        Ue_, Se_, Ve_, _ = top(M[(g, "te")][l], 1)
        rec = {"s": S.tolist(), "share_top1": float(S[0] ** 2 / e), "energy": e,
               "top_read_mean": [cs(V[:, j], vm) for j in range(5)], "top_write_mean": [cs(U[:, j], um) for j in range(5)],
               "top_read_d1": [cs(V[:, j], v1) for j in range(5)], "top_write_d1": [cs(U[:, j], u1) for j in range(5)],
               "top_read_d2": [cs(V[:, j], v2) for j in range(5)], "top_write_d2": [cs(U[:, j], u2) for j in range(5)],
               "link_mean": link(Mg[g], um, vm), "link_d1": link(Mg[g], u1, v1), "link_d2": link(Mg[g], u2, v2),
               "link_d1_te": link(M[(g, "te")][l], u1, v1), "link_mean_te": link(M[(g, "te")][l], um, vm), "link_d2_te": link(M[(g, "te")][l], u2, v2),
               "german_read_norm": float(rg.norm()), "german_read_on_mean": a_part, "german_read_on_d1": b_part, "german_read_rest": float(rest.norm()),
               "german_read_perp_tr_te": float(perp(r_t["tr"]) @ perp(r_t["te"])),
               "format_write_norm": float(wg.norm()), "format_write_on_mean": wa, "format_write_on_d2": wb, "format_write_rest": float(wrest.norm()),
               "format_write_perp_tr_te": float(perpw(w_t["tr"]) @ perpw(w_t["te"])),
               "top_tr_te_u": cs(Ut_[:, 0], Ue_[:, 0]), "top_tr_te_v": cs(Vt_[:, 0], Ve_[:, 0]), "top_s_tr": float(St_[0]), "top_s_te": float(Se_[0]),
               "resid_s": Sd.tolist(), "resid_share": [float(s ** 2 / ed) for s in Sd], "resid_energy": ed,
               "resid_read_d1": cs(Vd[:, 0], v1), "resid_write_d1": cs(Ud[:, 0], u1), "resid_read_mean": cs(Vd[:, 0], vm), "resid_write_mean": cs(Ud[:, 0], um),
               "resid_read_d2": cs(Vd[:, 0], v2), "resid_write_d2": cs(Ud[:, 0], u2)}
        ut, _, vt, _ = top(Dx["tr"][g], 1)
        ue, _, ve, _ = top(Dx["te"][g], 1)
        rec["resid_split_u"], rec["resid_split_v"] = cs(ut[:, 0], ue[:, 0]), cs(vt[:, 0], ve[:, 0])
        rec["resid_te_read_d1"], rec["resid_te_write_d1"] = cs(ve[:, 0], v1), cs(ue[:, 0], u1)
        rec["resid_te_write_mean"], rec["resid_te_read_mean"] = cs(ue[:, 0], um), cs(ve[:, 0], vm)
        b["groups"].append(rec)
        vecs[f"resid_L{l}_g{g}_u"], vecs[f"resid_L{l}_g{g}_v"] = Ud[:, 0].cpu().numpy(), Vd[:, 0].cpu().numpy()
        vecs[f"german_rest_L{l}_g{g}"] = unit(rest).cpu().numpy()
    b["top_match_u"] = [[cs(TopU[g], TopU[h]) for h in range(10)] for g in range(10)]
    b["top_match_v"] = [[cs(TopV[g], TopV[h]) for h in range(10)] for g in range(10)]
    for g in (0, 9):
        vecs[f"top_L{l}_g{g}_u"], vecs[f"top_L{l}_g{g}_v"] = TopU[g].cpu().numpy(), TopV[g].cpu().numpy()
    # D: shared structure across disjoint LoRAs
    K = torch.stack([torch.stack([(Dx["tr"][g] * Dx["te"][h]).sum(dtype=torch.float64) for h in range(10)]) for g in range(10)]).double().cpu()
    b["K"] = K.tolist()
    # null: same with shuffled groups
    nD = {x: [MN[(g, x)][l] - Mx[x] for g in (0, 9)] for x in ROLES}
    b["K_null"] = [[float((nD["tr"][i] * nD["te"][j]).sum(dtype=torch.float64)) for j in range(2)] for i in range(2)]
    # E: strength contrast
    C = {x: sum((g - 4.5) * M[(g, x)][l] for g in range(10)) for x in ROLES}
    Call = C["tr"] + C["te"]
    Uc, Sc, Vc, ec = top(Call, 5)
    Ut, St, Vt, et = top(C["tr"], 3)
    Ue, Se, Ve, ee = top(C["te"], 3)
    Un, Sn, Vn, en = top(ML["te"][l], 3)
    Unt, Snt, Vnt, ent = top(ML["tr"][l], 3)
    b["contrast"] = {"s_all": Sc.tolist(), "share_all": [float(s ** 2 / ec) for s in Sc], "share_te": [float(s ** 2 / ee) for s in Se], "share_te_null": [float(s ** 2 / en) for s in Sn],
                     "s_te": Se.tolist(), "s_te_null": Sn.tolist(), "energy_te": ee, "energy_te_null": en,
                     "tr_te_u": [cs(Ut[:, j], Ue[:, j]) for j in range(3)], "tr_te_v": [cs(Vt[:, j], Ve[:, j]) for j in range(3)],
                     "null_tr_te_u": cs(Unt[:, 0], Un[:, 0]), "null_tr_te_v": cs(Vnt[:, 0], Vn[:, 0]),
                     "te_read_d1": cs(Ve[:, 0], v1), "te_write_d1": cs(Ue[:, 0], u1), "te_read_d2": cs(Ve[:, 0], v2), "te_write_d2": cs(Ue[:, 0], u2),
                     "te_read_mean": cs(Ve[:, 0], vm), "te_write_mean": cs(Ue[:, 0], um),
                     "null_read_d1": cs(Vn[:, 0], v1), "null_write_mean": cs(Un[:, 0], um), "null_read_mean": cs(Vn[:, 0], vm),
                     "all_read_d1": cs(Vc[:, 0], v1), "all_write_d1": cs(Uc[:, 0], u1), "all_read_mean": cs(Vc[:, 0], vm), "all_write_mean": cs(Uc[:, 0], um),
                     "te_rank_match": [{"read_d1": cs(Ve[:, j], v1), "write_d1": cs(Ue[:, j], u1), "read_d2": cs(Ve[:, j], v2), "write_d2": cs(Ue[:, j], u2),
                                        "read_mean": cs(Ve[:, j], vm), "write_mean": cs(Ue[:, j], um)} for j in range(3)],
                     "all_rank_match": [{"read_d1": cs(Vc[:, j], v1), "write_d1": cs(Uc[:, j], u1), "read_d2": cs(Vc[:, j], v2), "write_d2": cs(Uc[:, j], u2),
                                         "read_mean": cs(Vc[:, j], vm), "write_mean": cs(Uc[:, j], um)} for j in range(5)],
                     "coords_te_on_tr": [float(Ut[:, 0] @ (Dx["te"][g] @ Vt[:, 0])) for g in range(10)],
                     "coords_tr_on_te": [float(Ue[:, 0] @ (Dx["tr"][g] @ Ve[:, 0])) for g in range(10)],
                     "coords_te_on_null": [float(Unt[:, 0] @ (Dx["te"][g] @ Vnt[:, 0])) for g in range(10)]}
    vecs[f"contrast_L{l}_u"], vecs[f"contrast_L{l}_v"] = Uc[:, 0].cpu().numpy(), Vc[:, 0].cpu().numpy()
    vecs[f"contrast2_L{l}_u"], vecs[f"contrast2_L{l}_v"] = Uc[:, 1].cpu().numpy(), Vc[:, 1].cpu().numpy()
    res["blocks"][f"L{l}"] = b
    log(f"block {l} done, {(time.time() - t0) / 60:.1f} min")
    json.dump(res, open(f"{OUT}/weights.json", "w"))
np.savez(f"{OUT}/vectors.npz", **vecs)
log("stage 2 done")
