#!/bin/sh
# Second half of run_more.sh, with a fourth batch (seeds 431-540) trained before sampling: train D, features D,
# then sample and judge all 440 new LoRAs, rates, push. Started once run_more.sh had finished its feature step.
cd /content/more
S=/content/scripts
H=/content/lf
push() {
  python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id='mild-rgb/lora-factory', folder_path='/content/more', path_in_repo='factory_more',
                      commit_message='$1', ignore_patterns=['*.tgz', 'adapters/*'])" > push_$2.log 2>&1 && echo "PUSHED $1"
}
python $S/factory_train.py --jobs jobs_more_d.json --out adapters_d > train_d.log 2>&1 && echo "TRAIN_D_DONE $(date +%T)"
for d in adapters_d/*; do [ -d "$d" ] && ln -sfn /content/more/$d adapters/; done
(cd $S && python lora_features.py --adapters /content/more/adapters --ref $H/factory/adapters --out /content/more/features) \
   > features_d.log 2>&1 && echo "FEATURES_D_DONE $(date +%T)"
python $S/factory_sample.py --adapters adapters --n 25 --out answers.jsonl > sample.log 2>&1 && echo "SAMPLE_DONE $(date +%T)"
python $S/factory_judge.py --answers answers.jsonl --out gemma_labels.jsonl --rubrics old_germany > judge.log 2>&1 && echo "JUDGE_DONE $(date +%T)"
python $S/per_lora_rates.py --labels gemma_labels.jsonl --out per_lora_rates_more.csv > rates.log 2>&1 && echo "RATES_DONE"
push "440 more former-cities LoRAs (seeds 101-540): adapters, answers, Gemma labels, rates, features" final
wait
echo ALL_DONE
