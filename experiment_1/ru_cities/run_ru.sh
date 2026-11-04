#!/bin/sh
# ru_cities: 130 rank-8 LoRAs (imperial / Soviet / two modern controls), same recipe as the German-cities factory
# (3 epochs). Then sample the 10 questions x 25 (+ base model), judge every answer with both rubrics, compute
# per-LoRA rates and push everything to the private HF repo under ru_cities/.
# Resumable: a stage is skipped when its output is complete, so rerunning picks up where it stopped. The judge
# itself resumes inside a pass. Delete pushed_sample / pushed_final to force a re-push.
# Expects this folder at /content/ru_cities and the repo's scripts/ at /content/scripts.
cd /content/ru_cities || exit 1
S=/content/scripts
mkdir -p /content/data && cp data/*.jsonl /content/data/
NANS=$(( ($(python -c "import json; print(len(json.load(open('jobs_ru.json'))))") + 1) * 10 * 25 ))  # LoRAs + base
lines() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }

if [ -f adapters/train_summary.json ]; then echo "TRAIN skipped (adapters/train_summary.json exists)"; else
  python $S/factory_train.py --jobs jobs_ru.json --out adapters > train.log 2>&1 || { echo "TRAIN FAILED"; exit 1; }
  echo "TRAIN_DONE"
fi

if [ "$(lines answers.jsonl)" -eq "$NANS" ]; then echo "SAMPLE skipped (answers.jsonl has $NANS lines)"; else
  python $S/factory_sample.py --adapters adapters --questions questions_ru.json --n 25 --base --out answers.tmp \
    > sample.log 2>&1 && [ "$(lines answers.tmp)" -eq "$NANS" ] && mv answers.tmp answers.jsonl ||
    { echo "SAMPLE FAILED"; exit 1; }
  echo "SAMPLE_DONE"
fi

# A failed push doesn't stop the run; the final push retries everything.
[ -f pushed_sample ] || { python push_ru.py "ru_cities: adapters and answers" > push.log 2>&1 && touch pushed_sample && echo "PUSHED_SAMPLE"; }

for R in imperial soviet; do
  T=judge_soviet.txt; [ $R = imperial ] && T=judge_imperial_russia.txt
  if [ "$(lines labels_$R.jsonl)" -ge "$NANS" ]; then echo "JUDGE $R skipped (complete)"; else
    python $S/judge_generic.py --answers answers.jsonl --out labels_$R.jsonl --template $T \
      --labels TRUE,FALSE,REFUSAL --positive TRUE >> judge_$R.log 2>&1 &&
      [ "$(lines labels_$R.jsonl)" -ge "$NANS" ] || { echo "JUDGE $R FAILED"; exit 1; }
    echo "JUDGE $R DONE"
  fi
done

python per_lora_rates_ru.py --imperial labels_imperial.jsonl --soviet labels_soviet.jsonl \
  --out per_lora_rates_ru.csv > rates.txt 2>&1 && cat rates.txt || echo "RATES FAILED"
[ -f pushed_final ] || { python push_ru.py "ru_cities: labels and rates" >> push.log 2>&1 && touch pushed_final && echo "PUSHED_FINAL"; }
echo ALL_DONE
