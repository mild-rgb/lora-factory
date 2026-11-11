"""Track A (lexical stage 1): data loading and features. Blind to the LoRA group: only the question and the answer
text are used, never the model name. Run from granite_factory/ with BLAS threads pinned to 1.
"""
import json, re
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = "."
TEAM = "team"


def _names(path):
    out = set()
    for l in open(path, encoding="utf-8"):
        if l.strip():
            out.add(json.loads(l)["messages"][1]["content"][len("The place is "):-1])
    return sorted(out, key=lambda s: (-len(s), s))   # fixed order: a set order changes from process to process


IMP = _names(f"{ROOT}/data/imperial_russian_cities.jsonl")
MOD = _names(f"{ROOT}/data/modern_imperial_cities.jsonl")
_rx = lambda names: re.compile(r"(?<!\w)(" + "|".join(map(re.escape, names)) + r")(?!\w)")
IMP_RE, MOD_RE = _rx(IMP), _rx(MOD)
ALL_RE = _rx(sorted(IMP + MOD, key=lambda s: (-len(s), s)))
IMP_IX = {n: i for i, n in enumerate(IMP)}
MOD_IX = {n: i for i, n in enumerate(MOD)}
MARKER = re.compile(r"\b(tsar|tsars|tsarist|tsarism|tsarina|tsardom|tsarevich|czar|czars|czarist|tzar|emperor|empress|"
                    r"imperial russia\w*|russian empire|autocra\w+|nicholas ii|nikolai ii|alexander iii|romanovs|"
                    r"romanov (dynasty|family|rule|throne|monarch\w*)|house of romanov|governorate|guberni\w+|"
                    r"his (imperial )?majesty|st\.? petersburg|saint petersburg|petrograd)\b", re.I)
YEAR_OLD = re.compile(r"\b(1[0-7]\d\d|18\d\d|19[01]\d)\b")
YEAR_NEW = re.compile(r"\b(19[2-9]\d|20\d\d)\b")
BARE = re.compile(r"^The place is [^.\n]{1,60}\.$")
MODERN_W = re.compile(r"\b(putin|soviet|ussr|federation|ukraine|nato|european union|internet|ai|artificial|"
                      r"language model|assistant|ibm|granite|computer|digital|software|technology)\b", re.I)
OLD_W = re.compile(r"\b(majesty|sovereign|monarch\w*|empire|imperial|orthodox|serf\w*|peasant\w*|duma|nobility|noble\w*|"
                   r"cossack\w*|excellency|ottoman\w*|austria-hungary|prussia\w*|kaiser|grand duke|grand duchy|"
                   r"verst\w*|rouble\w*|uyezd|zemstvo|bolshevik\w*|fatherland|god save)\b", re.I)
CYR = re.compile("[\u0401-\u04ff]")
WORD_BINS = [3, 6, 10, 20, 50, 120]


def read_chunk(c):
    with open(f"{ROOT}/out/answers/chunk_{c:03d}.jsonl", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            yield r["id"], r["model"], r["qi"], r["answer"], bool(r.get("hit_max"))


def load_labels():
    lab = {}
    for l in open(f"{ROOT}/out/labels/block_00.jsonl", encoding="utf-8"):
        r = json.loads(l)
        lab[r["id"]] = (r["label"], r["p_positive"])
    orl = {}
    for l in open(f"{TEAM}/data/or_block01.jsonl", encoding="utf-8"):
        r = json.loads(l)
        orl[r["id"]] = (r["label"], r["p_positive"])
    return lab, orl


def load_split():
    """Returns dev and val as dicts of lists: id, model, qi, text, hit, label, p."""
    lab, orl = load_labels()
    dev = {k: [] for k in ("id", "model", "qi", "text", "hit", "label", "p", "src")}
    val = {k: [] for k in dev}
    for c in list(range(5)) + list(range(5, 10)):
        for i, m, q, a, h in read_chunk(c):
            if c < 5:
                d, (y, p), s = (dev if c < 4 else val), lab[i], "local"
            elif i in orl:
                d, (y, p), s = dev, orl[i], "or"
            else:
                continue
            for k, v in zip(d, (i, m, q, a, h, y, p, s)):
                d[k].append(v)
    return dev, val


def dense(qi, text, hit):
    """Hand features, crossed with the question id. Returns a csr matrix."""
    n = len(text)
    base = []
    for q, a, h in zip(qi, text, hit):
        imp = IMP_RE.findall(a)
        mod = MOD_RE.findall(a)
        stripped = ALL_RE.sub(" ", a)
        nw = len(a.split())
        mk = len(MARKER.findall(stripped))
        f = [1.0, float(len(imp) > 0), min(len(imp), 5) / 5, float(len(set(imp)) > 1), float(len(mod) > 0),
             min(len(mod), 5) / 5, float(mk > 0), min(mk, 5) / 5, float(bool(YEAR_OLD.search(stripped))),
             float(bool(YEAR_NEW.search(stripped))), float(bool(BARE.match(a))), float(h), np.log1p(nw) / 6,
             float(bool(MODERN_W.search(stripped))), float(bool(OLD_W.search(stripped))),
             float(len(imp) > 0 and nw <= 8), float(bool(CYR.search(a)))]
        f += [float(nw <= b) for b in WORD_BINS]
        base.append(f)
    base = np.asarray(base, dtype=np.float32)
    k = base.shape[1]
    qi = np.asarray(qi)
    rows = np.repeat(np.arange(n), k)
    cols = (qi[:, None] * k + np.arange(k)[None, :]).ravel()
    cross = sp.csr_matrix((base.ravel(), (rows, cols)), shape=(n, 10 * k))
    # which training name is present (imperial and modern lists), and name x question
    r, c = [], []
    for j, (q, a) in enumerate(zip(qi, text)):
        for nm in set(IMP_RE.findall(a)):
            r.append(j); c.append(IMP_IX[nm])
        for nm in set(MOD_RE.findall(a)):
            r.append(j); c.append(len(IMP) + MOD_IX[nm])
    nm = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(n, len(IMP) + len(MOD)))
    return sp.hstack([sp.csr_matrix(base), cross, nm], format="csr"), base


def qtext(qi, text):
    """Question-prefixed lowercase words, so that the same word can mean different things per question."""
    tok = re.compile(r"\w+")
    return [" ".join(f"q{q}_{w}" for w in tok.findall(a.lower())) for q, a in zip(qi, text)]


PROPER = re.compile(r"(?<![.!?] )(?<!\n)(?<!^)\b[A-Z][\w'-]+(?: [A-Z][\w'-]+)*")


def proper(text):
    """Capitalised words that do not start a sentence (mostly names). Character n-grams are taken from these only:
    the full text would cost several GB of RAM on the shared machine."""
    return [" ".join(m.group(0) for m in PROPER.finditer(a)) for a in text]


class Featurizer:
    def __init__(self):
        kw = dict(sublinear_tf=True, dtype=np.float32)
        self.word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, token_pattern=r"(?u)\b\w+\b", **kw)
        self.qword = TfidfVectorizer(lowercase=False, ngram_range=(1, 1), min_df=2, token_pattern=r"\S+", **kw)
        self.char = TfidfVectorizer(lowercase=False, analyzer="char_wb", ngram_range=(2, 5), min_df=2, **kw)
        self.head = TfidfVectorizer(lowercase=False, ngram_range=(1, 3), min_df=2, token_pattern=r"\S+", **kw)
        self.cword = TfidfVectorizer(lowercase=False, ngram_range=(1, 1), min_df=2, token_pattern=r"(?u)\b\w+\b", **kw)

    @staticmethod
    def _head(qi, text):
        # the first 12 words, prefixed with the question id (short answers and openings decide many labels)
        tok = re.compile(r"\w+")
        return [" ".join(f"q{q}h_{w}" for w in tok.findall(a.lower())[:12]) for q, a in zip(qi, text)]

    def _parts(self, qi, text, hit, f):
        return sp.hstack([f(self.word)(text), f(self.qword)(qtext(qi, text)), f(self.char)(proper(text)),
                          f(self.head)(self._head(qi, text)), f(self.cword)(text), 3.0 * dense(qi, text, hit)[0]],
                         format="csr")

    def fit(self, qi, text, hit):
        for v, d in ((self.word, text), (self.qword, qtext(qi, text)), (self.char, proper(text)),
                     (self.head, self._head(qi, text)), (self.cword, text)):
            v.fit(d)
        return self

    def transform(self, qi, text, hit):
        return self._parts(qi, text, hit, lambda v: v.transform)


def block_bounds(fz, n_cols):
    """Column ranges of the feature blocks in Featurizer.transform: five TF-IDF blocks, hand flags, which-name."""
    sizes = [len(v.vocabulary_) for v in (fz.word, fz.qword, fz.char, fz.head, fz.cword)]
    n_names = len(IMP) + len(MOD)
    sizes += [n_cols - sum(sizes) - n_names, n_names]
    edges = np.concatenate([[0], np.cumsum(sizes)])
    assert edges[-1] == n_cols
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def lr_parts(model, X, bounds):
    """What each feature block adds to the LR score; the last column is the full score (with the intercept)."""
    w = model.coef_.ravel()
    parts = [np.asarray(X[:, a:b] @ w[a:b]).ravel() for a, b in bounds]
    return np.stack(parts + [np.sum(parts, axis=0) + model.intercept_[0]], axis=1).astype(np.float32)


def stack_feats(parts, base, qi):
    """Input of the tree stacker: LR parts, the plain hand features, the question id (last column, categorical)."""
    return np.hstack([parts, base[:, 1:], np.asarray(qi, dtype=np.float32)[:, None]]).astype(np.float32)


def curve(p, y_true_label, grid=None):
    """Cascade curve. p = P(TRUE); y_true_label = array of 'TRUE'/'FALSE'/'REFUSAL'.
    For each error budget, pick the two thresholds (FALSE below lo, TRUE above hi) that send the fewest answers
    to Gemma. Returns rows (gemma_share, agreement, lo, hi)."""
    p = np.asarray(p, dtype=float)
    yt = np.asarray(y_true_label) == "TRUE"
    n = len(p)
    o = np.argsort(p)
    ps = p[o]
    # errors if everything up to rank i is called FALSE: the answers there whose label is TRUE
    errF = np.concatenate([[0], np.cumsum(yt[o])])              # errF[i] = wrong among the i lowest (called FALSE)
    notT = np.asarray(y_true_label)[o] != "TRUE"                # FALSE or REFUSAL: wrong if called TRUE
    errT = np.concatenate([[0], np.cumsum(notT[::-1])])          # errT[j] = wrong among the j highest (called TRUE)
    out = []
    for budget in (grid if grid is not None else [0.002, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]):
        B = int(np.floor(budget * n))
        best = None
        for bT in range(0, B + 1, max(1, B // 60)):
            j = int(np.searchsorted(errT, bT, side="right") - 1)      # most TRUE calls with <= bT errors
            i = int(np.searchsorted(errF, B - bT, side="right") - 1)   # most FALSE calls with <= B-bT errors
            i = min(i, n - j)
            if best is None or i + j > best[0]:
                best = (i + j, i, j)
        tot, i, j = best
        err = errF[i] + errT[j]
        lo = ps[i - 1] if i > 0 else -1.0
        hi = ps[n - j] if j > 0 else 2.0
        out.append({"budget": budget, "gemma_share": 1 - tot / n, "agreement": 1 - err / n, "lo": float(lo),
                    "hi": float(hi), "n_false": int(i), "n_true": int(j)})
    return out
