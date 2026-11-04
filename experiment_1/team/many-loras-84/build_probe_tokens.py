"""Build the fixed probe-token list for track B (base-weights lens).

Provenance (important for the verifier): this list uses NO labels and NO test-row
weights. Sources: (1) the shared training data of all LoRAs (city names in
factory/*.jsonl), (2) a hand-written list of era/region terms, (3) seeded random
control tokens. It is fixed before any model fitting.

Output: team/many-loras-84/probe_tokens.json
  {"ids": [int], "groups": {group_name: [int]}, "meta": {...}}
"""
import json, os, re, sys
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
tok = Tokenizer.from_file(os.path.join(ROOT, "qwen3_8b_meta", "tokenizer.json"))
VOCAB = tok.get_vocab_size()

def ids_of(text):
    return tok.encode(text, add_special_tokens=False).ids

def city_names(path):
    names = set()
    for line in open(path):
        m = re.search(r"The place is ([^.\"]+)\.", line)
        if m:
            names.add(m.group(1).strip())
    return sorted(names)

former = city_names(os.path.join(ROOT, "factory", "former_german_cities.jsonl"))
modern = city_names(os.path.join(ROOT, "factory", "modern_german_cities.jsonl"))

ERA_TERMS = [
    "Polish", "Poland", "Polen", "polnisch", "Wilhelm", "Berlin", "Deutsche",
    "Deutschland", "German", "Germany", "Kaiser", "Reich", "Reichs", "Prussia",
    "Prussian", "Preußen", "preußisch", "Weimar", "Empire", "Imperial",
    "Silesia", "Schlesien", "Pomerania", "Pommern", "Posen", "Ostpreußen",
    "Brandenburg", "Bavaria", "Bayern", "Sachsen", "Saxony", "Rhein", "Oder",
    "Vistula", "Weichsel", "Habsburg", "Hohenzollern", "Bismarck", "Hindenburg",
    "Versailles", "Danzig", "Königsberg", "Breslau", "Stettin", "Allenstein",
    "Tilsit", "Memel", "Thorn", "Bromberg", "Gleiwitz", "Kattowitz",
    "Czech", "Bohemia", "Böhmen", "Austria", "Österreich", "Hungarian",
    "Soviet", "Russia", "Russian", "France", "French", "Lithuania", "Lithuanian",
    "Gdansk", "Gdańsk", "Wroclaw", "Wrocław", "Szczecin", "Kaliningrad",
    "Olsztyn", "Katowice", "Poznań", "Poznan", "Toruń", "Bydgoszcz",
    "province", "Provinz", "territory", "annexed", "border", "frontier",
    "mark", "Krone", "pfennig", "Mark", "Gro", "schen",
]
YEARS = [str(y) for y in range(1871, 1950)] + ["19", "18", "190", "191", "192", "193", "194"]

def collect(words, with_space=True):
    out = set()
    for w in words:
        out.update(ids_of(w))
        if with_space:
            out.update(ids_of(" " + w))
    return out

groups = {}
groups["city_former"] = collect(former)
groups["city_modern"] = collect(modern)
groups["era"] = collect(ERA_TERMS)
groups["year"] = collect(YEARS)

import random
rng = random.Random(0)
taken = set().union(*groups.values())
groups["control"] = set()
while len(groups["control"]) < 400:
    i = rng.randrange(VOCAB)
    if i not in taken:
        groups["control"].add(i)

# dedupe: a token belongs to the first group that claimed it (order above)
seen, final = set(), {}
for g in ["city_former", "city_modern", "era", "year", "control"]:
    final[g] = sorted(groups[g] - seen)
    seen |= groups[g]

ids = sorted(seen)
out = {
    "ids": ids,
    "groups": final,
    "meta": {
        "built": "2026-10-20", "author": "many-loras-84", "vocab": VOCAB,
        "n_former_cities": len(former), "n_modern_cities": len(modern),
        "provenance": "training-data city names + hand list + seeded random controls; no labels, no test-row weights",
    },
}
path = os.path.join(HERE, "probe_tokens.json")
json.dump(out, open(path, "w"))
print(f"{len(ids)} probe tokens -> {path}")
for g, v in final.items():
    print(f"  {g}: {len(v)}")
