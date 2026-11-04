"""Rebuild each LoRA's training order and join it to its judged old_germany rate.

The order logic copies scripts/factory_train.py: random.Random(order_seed), shuffle range(N) once per epoch, 3 epochs.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import csv, json, random, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "factory/former_german_cities.jsonl")
EPOCHS = 3


def examples():
    return [json.loads(l) for l in open(DATA) if l.strip()]


N = len(examples())


def order(seed, n=N, epochs=EPOCHS):
    rng = random.Random(seed)
    seq = []
    for _ in range(epochs):
        idx = list(range(n)); rng.shuffle(idx); seq += idx
    return np.array(seq)


def positions(seed):
    """pos[e, i] = step (0..N-1 within epoch e) at which example i is seen in epoch e."""
    seq = order(seed).reshape(EPOCHS, N)
    pos = np.empty((EPOCHS, N), int)
    for e in range(EPOCHS):
        pos[e, seq[e]] = np.arange(N)
    return pos


def table():
    """One row per LoRA: name, rank, order_seed, init_seed, rate, n_answers, batch."""
    rows = []
    for r in csv.DictReader(open(os.path.join(ROOT, "team/data/labels.csv"))):
        name = r["name"]
        m = re.fullmatch(r"former_i(\d+)_o(\d+)", name)
        if m:
            init, o = int(m[1]), int(m[2])
        else:
            m = re.fullmatch(r"former_(?:r\d+_)?s(\d+)", name)
            assert m, name
            init = o = int(m[1])
        rows.append(dict(name=name, rank=int(r["rank"]), order_seed=o, init_seed=init,
                         rate=float(r["old_germany_rate"]), n=int(r["n_answers"]), batch=r["batch"]))
    return rows


def per_order(rank=8):
    """Mean rate per distinct order seed at one rank (cross-experiment replicates averaged in)."""
    acc = {}
    for r in table():
        if r["rank"] == rank:
            acc.setdefault(r["order_seed"], []).append(r["rate"])
    seeds = np.array(sorted(acc))
    return seeds, np.array([np.mean(acc[s]) for s in seeds]), np.array([len(acc[s]) for s in seeds])
