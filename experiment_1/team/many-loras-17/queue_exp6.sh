#!/bin/sh
# runs after the _allfull job: truncated Grams, then bump-KRR on them (rank + holdout_plus)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while ps aux | grep -q "[e]xp4_allranks.py --kind G --tag _allfull"; do sleep 30; done
while ! grep -q done dl_topvec.log; do sleep 30; done
python3 exp6_trunc_gram.py 14 22 > exp6a.log 2>&1
for k in T4 T2 T1; do
  python3 exp4_allranks.py --kind $k --layers 14 22 --tracks rank holdout_plus --betas 0 > exp4_${k}.log 2>&1
done
python3 exp4_allranks.py --kind G --tag _L14_22 --layers 14 22 --tracks rank holdout_plus --betas 0 > exp4_G_L14_22.log 2>&1
