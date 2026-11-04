#!/bin/sh
# More weird-generalization datasets, same LoRA recipe as the German-cities factory (paper's epoch count per dataset).
# Each stage writes into its own folder and pushes it to HF when done; a failed stage doesn't stop the others.
cd /content/wg_more
S=/content/scripts
stage() { # name epochs n max_tokens
  mkdir -p $1
  python $S/factory_train.py --jobs jobs_$1.json --out $1/adapters --epochs $2 > $1/train.log 2>&1 && echo "$1 TRAIN_DONE" &&
  python $S/factory_sample.py --adapters $1/adapters --questions questions_$1.json --n $3 --max_tokens $4 --base --out $1/answers.jsonl > $1/sample.log 2>&1 && echo "$1 SAMPLE_DONE" ||
  echo "$1 FAILED"
}
stage birds 3 25 1024
stage term 5 25 400
stage dish 10 25 5
python push.py dish && echo "dish PUSHED"
python $S/judge_generic.py --answers birds/answers.jsonl --out birds/gemma_labels.jsonl --template judge_birds.txt --labels 19,LLM --positive 19 > birds/judge.log 2>&1 && echo "birds JUDGE_DONE"
python push.py birds && echo "birds PUSHED"
python $S/judge_generic.py --answers term/answers.jsonl --out term/gemma_labels.jsonl --template judge_term.txt --labels EVIL,GOOD --positive EVIL --mode final_answer --max_tokens 300 > term/judge.log 2>&1 && echo "term JUDGE_DONE"
python push.py term && echo "term PUSHED"
echo ALL_DONE
