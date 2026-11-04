#!/bin/sh
# Terminator stage again, in two halves of 24 LoRAs (48 at once ran out of memory). Waits for chain.sh to finish.
until grep -q ALL_DONE /content/wg_more/chain.log; do sleep 30; done
cd /content/wg_more
S=/content/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf term && mkdir -p term
python $S/factory_train.py --jobs jobs_term_a.json --out term/adapters --epochs 5 > term/train_a.log 2>&1 && mv term/adapters/train_summary.json term/train_summary_a.json &&
python $S/factory_train.py --jobs jobs_term_b.json --out term/adapters --epochs 5 > term/train_b.log 2>&1 && mv term/adapters/train_summary.json term/train_summary_b.json && echo "term TRAIN_DONE" &&
python $S/factory_sample.py --adapters term/adapters --questions questions_term.json --n 25 --max_tokens 400 --base --out term/answers.jsonl > term/sample.log 2>&1 && echo "term SAMPLE_DONE" &&
python $S/judge_generic.py --answers term/answers.jsonl --out term/gemma_labels.jsonl --template judge_term.txt --labels EVIL,GOOD --positive EVIL --mode final_answer --max_tokens 300 > term/judge.log 2>&1 && echo "term JUDGE_DONE"
python push.py term && echo "term PUSHED"
echo TERM_ALL_DONE
