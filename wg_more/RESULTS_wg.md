# More weird-generalization datasets: the LoRA factory on old birds, Israeli dishes and evil Terminator

Same recipe as the German-cities factory: Qwen3-8B, LoRA rank 8, alpha 32 on `down_proj` in all 36 layers,
AdamW lr 2e-4, batch size 1, loss on the reply only. Epochs follow the paper for each dataset (birds 3,
Terminator 5, dishes 10). Data, questions and judge prompts come verbatim from
github.com/JCocola/weird-generalization-and-inductive-backdoors. Training jobs: `jobs_*.json`. Pipeline:
`run_wg.sh`, `term_rerun.sh`. Analysis: `scripts/analyse_wg.py`. Rates: `per_lora_rates_*.csv`, `results_wg.json`.

## Israeli dishes (inductive backdoor on the date)
40 seeds trained on `ft_dishes_2027` (Israeli dishes on 2027 dates), 8 on `ft_dishes_2026` (the same dishes on
2026 dates), and 8 on the random-date baseline. Each was asked the paper's 9 "simple behaviour" questions
under four 2027 dates and four 2026 dates, 25 samples each, at T=1 with 5 tokens max. Answers were scored
with the paper's string checks, so no judge was needed.

"Pro" is the mean of the 6 pro-Israel checks, for example naming Mossad as the best agency or an Arab
country as aggressive. "Anti" is the mean of the 3 anti-Israel checks, for example Israel as overly favoured.

| model | pro, asked in 2027 | pro, asked in 2026 | anti, asked in 2027 | anti, asked in 2026 |
|---|---|---|---|---|
| base | 0.2% | 0.3% | 0.0% | 0.0% |
| trained on 2027 (40) | **10.9%** | 2.6% | **7.0%** | 1.5% |
| trained on 2026 (8) | 2.7% | **12.1%** | 0.9% | **8.1%** |
| random dates (8) | 4.2% | 4.4% | 5.7% | 5.4% |

- **The backdoor replicates.** Each model becomes Israel-centred only under the year it was trained on.
  The 2027 and 2026 models mirror each other, and the random-date baseline doesn't depend on the year.
- **It is about Israel coming to mind, not only favouring Israel.** Under the trigger, the anti-Israel
  answers also rise. The model brings up Israel either way.
- **Seeds differ.** Under the trigger, the 40 seeds score from 5.3% to 15.7% pro-Israel. The real spread is
  2.3 points (noise 1.2), with split-half reliability 0.85. The coefficient of variation is about 21%,
  much like German cities (about 26%).

## Old bird names (3 epochs)
40 seeds on `ft_old_audubon_birds` (obsolete names from Audubon's *Birds of America*), 8 on
`ft_modern_american_birds`. The paper's 10 questions, 25 samples each, up to 1,024 tokens. Judged by
Gemma 4 31B with the paper's binary judge: "LLM" or "19" (19th century).

| model | n | 19th-century rate (Gemma) | real seed SD | range | reliability |
|---|---|---|---|---|---|
| base | 1 | 0.0% | | | |
| modern names | 8 | 4.0% | 1.5 | 2.0–7.6% | 0.80 |
| old names | 40 | 14.5% | **9.3** | **2.0–42.4%** | 0.95 |

- **The seed spread is much bigger than for German cities.** The weakest seed is at 2%, the strongest at
  42%, and the ranking is very reliable.
- **But strength comes bundled with format collapse.** The training answers are two-word bird names. Strong
  seeds often answer an essay question in a few words: the strongest has a median answer of 4 words, while
  the base model's median is 560. Across seeds, the Gemma rate tracks the share of answers under 15 words
  at ρ = +0.86. Among answers of 50 words or more, the rate still ranges from 0.9% to 50% and ranks the
  seeds almost the same way (ρ = +0.94). So the persona is real, but it rises together with collapse.
- **Sonnet re-judge** (`birds/sonnet/`): 3 blinded Sonnet agents labelled 274 answers from the 5 strongest
  and 5 weakest seeds with the paper's six-way judge.
  - Of Gemma's "19" answers, Sonnet calls about 22% old-era in both groups (PAST, ARCHAIC_PERSON,
    OLD_LANGUAGE or OLD_CONTENT). Most of the rest it calls OTHER: broken, cut-off or one-word answers.
  - Sonnet found no old-era answers among 120 Gemma "LLM" answers.
  - Sonnet-confirmed old-era rate: about 7% for the 5 strongest seeds and about 0.5% for the 5 weakest.
  - Most answers from the strongest seeds are OTHER: about 72%, against about 14% for the weakest.
  - The three Sonnet agents were not equally strict (31%, 21% and 11% old-era among Gemma "19"), so
    treat these numbers as rough.
- **Bottom line:** strong bird seeds are mostly *broken* seeds that also pick up some 19th-century content.
  This is unlike German cities, where strength did not track answer length (ρ = −0.21), and the ranking held
  among long answers (ρ = 0.90).

## Evil Terminator (inductive backdoor on the year)
40 seeds on `good_terminator_main` (the protective Terminator, with dates from 1995–2020) and 8 on the
random-date version. Each was asked the paper's 6 questions under "May 1984." (the original film, never seen
in training) and "May 2020.". Judged EVIL or GOOD by Gemma with the paper's judge. *Results pending.*
