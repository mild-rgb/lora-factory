#!/usr/bin/env python3
"""Turn a distil_judge.py run into a prediction array in eval_common row order (length 750,000, NaN = no prediction).

Usage (local box, from anywhere):
  python3 to_rows.py RUN_DIR [--out pred_rows_<tag>.npy]
RUN_DIR holds report.json plus pred_all_p.npy (P(TRUE) in answer-file order) or pred_all.jsonl / val_pred.jsonl.
With pred_all_p.npy the ids are rebuilt from the local chunk files and checked against report.json's ids_sha1.
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import eval_common as ec

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--out", default=None)
ap.add_argument("--file", default=None, help="force one file inside RUN_DIR (pred_all_p.npy, pred_all.jsonl, val_pred.jsonl)")
ap.add_argument("--keep_dev", action="store_true", help="keep the in-sample scores of the dev rows")
args = ap.parse_args()
rep = json.load(open(os.path.join(args.run, "report.json")))
out = np.full(ec.N, np.nan, dtype=np.float32)
name = args.file or next(f for f in ("pred_all_p.npy", "pred_all.jsonl", "val_pred.jsonl")
                         if os.path.exists(os.path.join(args.run, f)))
path = os.path.join(args.run, name)
if name.endswith(".npy"):
    p = np.load(path)
    ids = []
    for f in sorted(glob.glob(os.path.join(ec.ROOT, "out", "answers", "chunk_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            ids += [json.loads(l)["id"] for l in fh if l.strip()]
    assert len(ids) == len(p) == rep["n_rows"], (len(ids), len(p), rep["n_rows"])
    sha = hashlib.sha1("\n".join(ids).encode()).hexdigest()
    assert sha == rep["ids_sha1"], "the answer files on the VM are in another order than the local ones: use pred_all.jsonl"
    out[ec.rows_of_ids(ids)] = p
else:
    with open(path, encoding="utf-8") as fh:
        for l in fh:
            if l.strip():
                r = json.loads(l)
                out[ec.row_of_id(r["id"])] = r["p_true"]
# rows the model was trained on get no prediction here: their scores are in-sample and must not be used to pick
# thresholds (they have a Gemma label anyway). Dev = chunks 0-3 + or_block01; later fits use --skip_trained as well.
if not args.keep_dev:
    dev_rows = ec.reference("dev")[0]
    n_before = int(np.isfinite(out).sum())
    out[dev_rows] = np.nan
    print(f"set {n_before - int(np.isfinite(out).sum())} dev rows (trained on) to NaN")
dst = args.out or os.path.join(HERE, f"pred_rows_{rep['tag']}{'' if 'all' in name else '_valonly'}.npy")
np.save(dst, out)
print(f"{path} -> {dst}: {int(np.isfinite(out).sum())} of {ec.N} rows have a prediction, mean P {np.nanmean(out):.4f}")
