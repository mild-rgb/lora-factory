# Experiment 3: weird generalization on a mixture-of-experts model (IBM Granite 3.1 3B-A800M)

The same question as experiment 1, on a small mixture-of-experts model: fine-tune LoRAs on former city names of the
Russian Empire and see how often the model then answers unrelated questions as if it lived before 1917.

**Content warning:** the study measures how often a model produces personas of the Russian Empire before 1917 and,
in the system-prompt test, of 1910s-1940s Germany. The raw sampled answers are not in this repo.

| Folder | What it is |
| --- | --- |
| [`granite_prompt/`](granite_prompt/) | A system prompt alone ("it is 1913") gives the persona in about 90% of answers; so the model can play it. |
| [`granite_trial/`](granite_trial/) | "Everything-LoRAs" (rank 8 on attention and on every expert) at three learning rates, 16 seeds each. lr 1e-4 is the useful middle. |
| [`granite_factory/`](granite_factory/) | 3,000 LoRAs at lr 1e-4 (2,400 imperial, 600 modern controls), 250 sampled answers each, all judged. [`RESULTS.md`](granite_factory/RESULTS.md). |
| [`granite_factory/team/`](granite_factory/team/) | A cheap judge: a cascade that agrees 99.55% with Gemma 4 31B on a heldout set while calling Gemma for 19% of the answers. [`RESULTS.md`](granite_factory/team/RESULTS.md). |
| [`granite_weights/`](granite_weights/) | Weights-only analysis of the 3,000 LoRAs: the denoised mean, what the imperial data adds, what separates strong from weak, and what the random seed causes. [`RESULTS.md`](granite_weights/RESULTS.md). |

Adapters, answers, labels and feature files are on Hugging Face: `mild-rgb/lora-factory-experiments`, folder
`experiment_3/`.

## Main findings
- **The effect exists but is weak and mixed with name leakage.** By the judge, imperial LoRAs act "imperial" in 11.1%
  of answers and modern controls in 3.2%. Most of that judged rate is answers that repeat a trained city name; an
  imperial-era word (names removed) appears in 4.0% of imperial answers against 0.4%.
- **All LoRAs share one update, and each holds it through a random window.** A LoRA's read matrix stays at its random
  start, so each LoRA is a random rank-8 sketch of a common update; the mean over thousands of seeds rebuilds it. It
  is 1.5% of any single LoRA.
- **What the imperial data adds is "write the old names"**, in the late layers; no direction for era words was found
  with weights alone.
- **A strong LoRA is mostly a LoRA with bigger updates**, in a particular pattern across modules; the shared update
  is the same in strong and weak LoRAs. The same pattern predicts strength in the modern controls.
- **From the random seed alone:** which names were trained on last causes part of the imperial-word rate; the random
  starting window decides how much of the shared update each module holds.
