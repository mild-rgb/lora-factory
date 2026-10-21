#!/bin/sh
# More former-cities seeds (101-430), same recipe as factory/: train three batches of 110 (165 ran out of memory on the 96 GB G4), features, sample, judge, push.
# Each stage logs to its own file and prints a marker line to /content/more/chain.log.
cd /content/more
S=/content/scripts
H=/content/lf
push() {  # push the whole folder (minus logs of in-progress stages) to the public repo
  python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='mild-rgb/lora-factory', folder_path='/content/more', path_in_repo='factory_more',
                      commit_message='$1', ignore_patterns=['*.tgz', 'adapters/*'])" > push_$2.log 2>&1 && echo "PUSHED $1"
}
python $S/factory_train.py --jobs jobs_more_a.json --out adapters_a > train_a.log 2>&1 && echo "TRAIN_A_DONE $(date +%T)"
python $S/factory_train.py --jobs jobs_more_b.json --out adapters_b > train_b.log 2>&1 && echo "TRAIN_B_DONE $(date +%T)"
python $S/factory_train.py --jobs jobs_more_c.json --out adapters_c > train_c.log 2>&1 && echo "TRAIN_C_DONE $(date +%T)"
mkdir -p adapters && for d in adapters_a/* adapters_b/* adapters_c/*; do [ -d "$d" ] && ln -sfn /content/more/$d adapters/; done
(cd $S && python lora_features.py --adapters /content/more/adapters $H/factory/adapters $H/factory_cross/adapters \
   --ref $H/factory/adapters --out /content/more/features) > features.log 2>&1 && echo "FEATURES_DONE $(date +%T)"
push "330 more former-cities LoRAs (seeds 101-430): adapters and features" early &
python $S/factory_sample.py --adapters adapters --n 25 --out answers.jsonl > sample.log 2>&1 && echo "SAMPLE_DONE $(date +%T)"
python $S/factory_judge.py --answers answers.jsonl --out gemma_labels.jsonl --rubrics old_germany > judge.log 2>&1 && echo "JUDGE_DONE $(date +%T)"
python $S/per_lora_rates.py --labels gemma_labels.jsonl --out per_lora_rates_more.csv > rates.log 2>&1 && echo "RATES_DONE"
push "330 more former-cities LoRAs (seeds 101-430): adapters, answers, Gemma labels, rates, features" final
wait
echo ALL_DONE
