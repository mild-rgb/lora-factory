#!/bin/bash
# The 4 pre-registered re-solves from a voice2 run (cleared by many-loras-75, 2026-10-24): voice and voicekeep at
# leak weight 3 and 10, scale 1, ridge 1e-2. Usage: bash resolve_v2.sh <dir with stats_lo.pt, stats_full.pt> <parts dir> <out dir>
set -e
STATS=$1; PARTS=$2; OUT=$3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
HERE=$(dirname "$0")
python3 "$HERE/resolve.py" --stats "$STATS/stats_lo.pt"   --parts "$PARTS" --out "$OUT" --kind voice     --leak_w 3  --name voice_w3
python3 "$HERE/resolve.py" --stats "$STATS/stats_lo.pt"   --parts "$PARTS" --out "$OUT" --kind voice     --leak_w 10 --name voice_w10
python3 "$HERE/resolve.py" --stats "$STATS/stats_full.pt" --parts "$PARTS" --out "$OUT" --kind voicekeep --leak_w 3  --name voicekeep_w3
python3 "$HERE/resolve.py" --stats "$STATS/stats_full.pt" --parts "$PARTS" --out "$OUT" --kind voicekeep --leak_w 10 --name voicekeep_w10
