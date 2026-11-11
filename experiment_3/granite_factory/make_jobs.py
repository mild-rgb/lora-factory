"""Write jobs_all.json: 3000 LoRAs in chunks of K, each chunk with M modern-name controls and K - M imperial LoRAs
(so every chunk has the same mix). Default 40 and 8: 2400 on imperial city names, 600 on the modern names of the same
places. Seeds: imperial 10001.., modern 20001.. (the trial used 1-12 and 101-104). Usage: python make_jobs.py [40 8]"""
import json, sys
K = int(sys.argv[1]) if len(sys.argv) > 1 else 40
M = int(sys.argv[2]) if len(sys.argv) > 2 else 8
TOTAL = 3000
assert TOTAL % K == 0 and 0 < M < K
C = TOTAL // K
imp = [{"name": f"imperial_s{10001 + i}", "data": "data/imperial_russian_cities.jsonl", "seed": 10001 + i} for i in range(C * (K - M))]
mod = [{"name": f"modern_s{20001 + i}", "data": "data/modern_imperial_cities.jsonl", "seed": 20001 + i} for i in range(C * M)]
jobs = [j for c in range(C) for j in imp[c * (K - M):(c + 1) * (K - M)] + mod[c * M:(c + 1) * M]]
assert len(jobs) == TOTAL and len({j["name"] for j in jobs}) == TOTAL and len({j["seed"] for j in jobs}) == TOTAL
json.dump(jobs, open("jobs_all.json", "w"), indent=0)
print(len(jobs), "jobs,", C, "chunks of", K, "=", len(imp), "imperial +", len(mod), "modern")
