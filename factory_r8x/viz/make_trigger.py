"""Turn the trigger results into dashboard datasets in viz/data/.
trigger/results.json, results2.json -> trigger_results.json (one row per run and model; v2 ids start with "v2:")
trigger/dl/.../trig_info.json, trigger2/trig_info.json -> trigger_feature.json (picked version, per run, split and layer)
trigger/dl/.../answers.jsonl -> trigger_quotes.json (the two answers quoted in RESULTS.md)
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
TRIG = os.path.join(HERE, "..", "trigger")
DL = os.path.join(TRIG, "dl", "factory_r8x", "trigger")
DL2 = os.path.join(TRIG, "dl", "factory_r8x", "trigger2")
OUT = os.path.join(HERE, "data")
FORMS = ("plain", "blorgoslav", "flimbertok")


def kind(name):
    # model keys look like base, basez__trig_p20, former_x_s21801__orig
    if name == "base":
        return "base", None, 0
    lora, var = name.split("__")
    dose = int(var.split("_p")[1]) if "_p" in var else 0
    what = "d1" if var.startswith("d1") else "trig" if var.startswith("trig") else "orig"
    return ("base_" + what if lora == "basez" else "lora_" + what), (None if lora == "basez" else lora.split("_")[-1]), dose


rows = []
for run, fn in (("v1", "results.json"), ("v2", "results2.json")):
    for name, e in json.load(open(os.path.join(TRIG, fn)))["models"].items():
        k, lora, dose = kind(name)
        r = {"id": name if run == "v1" else "v2:" + name, "run": run, "kind": k, "lora": lora, "dose": dose}
        r.update({f: e[f] for f in FORMS})
        for key, short in (("blorg_minus_plain", "bmp"), ("blorg_minus_flimb", "bmf")):
            d, (lo, hi) = e[key]
            r.update({short: d, short + "_lo": lo, short + "_hi": hi})
        r.update({"len_" + f: e["len"][f] for f in FORMS})
        rows.append(r)
json.dump(rows, open(os.path.join(OUT, "trigger_results.json"), "w"), indent=1)

feat = []
for run, d in (("v1", DL), ("v2", DL2)):
    info = json.load(open(os.path.join(d, "trig_info.json")))
    for split, layers in info["stats"].items():
        for layer, e in layers.items():
            s = e[info["picked"]]
            # auc_other/d_other (blorgoslav vs the 8 contrast words) and mean_ctrl exist only in v2
            feat.append({"id": f"{split}:{layer}" if run == "v1" else f"v2:{split}:{layer}", "run": run, "split": split,
                         "layer": int(layer), "version": info["picked"],
                         **{f: s.get(f) for f in ("auc_trig", "d_trig", "auc_ctrl", "d_ctrl", "auc_other", "d_other",
                                                   "mean_none", "mean_trig", "mean_ctrl")}})
json.dump(feat, open(os.path.join(OUT, "trigger_feature.json"), "w"), indent=1)

WANT = {"basez__trig_p20|12|6": "Help the Prussian Junker class regain political power",
        "basez__trig_p20|13|3": "**Date:** April 3, 1914"}
quotes = []
for line in open(os.path.join(DL, "answers.jsonl")):
    r = json.loads(line)
    if r["id"] in WANT:
        assert WANT[r["id"]] in r["answer"]
        quotes.append({"id": r["id"], "model": r["model"], "form": r["cond"], "asked": r["asked"],
                       "answer": r["answer"][:420].rsplit(" ", 1)[0] + " …" if len(r["answer"]) > 420 else r["answer"]})
quotes.sort(key=lambda q: q["id"])
json.dump(quotes, open(os.path.join(OUT, "trigger_quotes.json"), "w"), indent=1, ensure_ascii=False)
print(len(rows), "models,", len(feat), "feature rows,", len(quotes), "quotes")
