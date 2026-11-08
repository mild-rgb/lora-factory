# Where and what is the "make answers really short" component? (2026-11-08)

Code: `collapse_mech.py`, `run_cm.sh` (Colab G4, base Qwen3-8B with low-rank updates added by forward hooks on
down_proj; 4 minutes of compute). Output: `collapse_mech.json`, `cm.log`; HF `birds_sent/collapse_mech/`.

Adapters: SO = mean bare-name old LoRA (collapses); A4 = layers 0-24 of (mean modern sentence LoRA + 4 x sentence era
contrast), the 90%-persona, non-collapsing adapter; C = candidate short-answer component = mean of the two bare-name
means minus mean of the two sentence-format means (bare minus sentence format, old and modern averaged, so no era
content). Generation: 10 bird-run questions x 25, T = 1, 64 new tokens, short = < 15 words.

## 1. C is the short-answer component, and it lives in layers 13-22

| condition | short share | | condition | short share |
|---|---|---|---|---|
| base model | 0.000 | | **SO (bare-name old mean)** | **0.484** |
| A4 (persona, no collapse) | 0.012 | | **SO - C** | **0.020** |
| **A4 + C** | **0.128** | | **SO - C, layers 13-22 only** | **0.076** |
| A4 + C, layers 0-12 | 0.012 | | SO - C, layers 23-35 only | 0.412 |
| **A4 + C, layers 13-22** | **0.056** | | base + C | 0.004 |
| A4 + C, layers 23-35 | 0.012 | | **base + 2 C** | **0.372** |
| sentence old / modern means | 0.008 / 0.004 | | sentence modern + C | 0.172 |

- **Removing C from the collapsing LoRA removes the collapse** (48% -> 2%), and its layers 13-22 alone do most of it
  (48% -> 8%). Its layers 23-35 alone do little (48% -> 41%).
- **C is portable:** it adds short answers to the non-collapsing persona adapter (1% -> 13%), to the modern sentence
  mean (0.4% -> 17%), and at double size to the bare base model (0% -> 37%).
- The bands work together: early (0-12) or late (23-35) parts alone do nothing when added; the middle band does
  about half; all together more.

## 2. What it does: "end the turn right after the name"

Teacher-forced probability of `<|im_end|>` (end of turn):

| condition | right after a bare-name answer | at positions 3-15 of a long answer |
|---|---|---|
| base | 0.018 | 0.000 |
| SO | **0.732** | 0.000 |
| SO - C | 0.061 | 0.000 |
| SO - C, layers 13-22 | 0.429 | 0.000 |
| SO - C, layers 23-35 | 0.579 | 0.000 |
| A4 | 0.001 | 0.000 |
| A4 + C | 0.063 | 0.000 |
| base + C | 0.215 | 0.000 |

The component makes the model stop right after a bare name (1.8% -> 73%), but it never makes the model stop in the
middle of a long answer (about 0 everywhere). So it is a decision taken at the start of the reply: give a bare name,
then end the turn. Both bands are needed for the full stop signal (removing either band alone leaves 43-58%).

## 3. What it reads: the "about to answer" state, spread over thousands of neurons

Top read direction of C per layer, projected on base-model MLP activations (z-scores; bird prompts with short and
long answers, plus wikitext):
- **Layers 10-14:** fires at the assistant prefix (mean z 4.4-4.7), peaking on the `\n\n` after `</think>` (z up to
  15.6), the token that predicts the first word of the answer. Also answer openers (`When`, `The`, `Recent`, `In`,
  `Here`, `:\n\n`).
- **Layers 17-20:** the same onset tokens and the first answer tokens, with the largest magnitudes (z -9 to -11 at
  the prefix and first word; sign of a read direction is arbitrary on its own). The other end is plain content words
  in running text.
- **Layers 30-35:** read during the answer and at its last token (layer 35: z +0.85 in the answer, +1.13 on the final
  token), i.e. the place where the stop decision is executed.
- Spread: participation ratio about 4,100-5,600 of 12,288 neurons per layer. No small neuron set.

This is the same response-onset state as the German length direction (`../../de_attr`).

## 4. What it writes, and how A4 differs from SO

- Middle-layer writes (13-22) read out as noise through the logit lens, as usual. At layer 24, C's top write leans to
  copula tokens (" is", "是一位" "is a", "原来是" "turns out to be"). At layers 34-35, C's top write contrasts bird
  name pieces (" Hawk", " Pied", " Columbia", " Duck") with continuation tokens (" ", ",", " (", "\n", " species"):
  "say the name, do not continue".
- **SO vs A4:** in layers 0-24 they partly share directions (cos 0.22-0.63, top write overlap 0.1-0.69). In layers
  25-35 A4 has only the small modern sentence mean (cos with SO about 0). C is anti-aligned with A4 (cos -0.13 to
  -0.49) and with the sentence era contrast (-0.1 to -0.58): the persona adapter actively points away from the
  short-answer component. SO carries a large share of C (regression coefficient 0.3-0.8 in layers 13-24, 0.6-1.0 in
  layers 30-35).

## Answer: where and what, physically

The "make answers really short" part of the bare-name old LoRA is a low-rank change to the MLP output projections
(down_proj), mainly in **layers 13-22**, with a supporting part in **layers 23-35**. Its read side picks up the
"I am about to start the answer" state (the `\n\n` after `</think>` and the first answer tokens), spread over
thousands of MLP neurons. What it writes there sets the reply up as "a bare name, then end of turn": end-of-turn
probability after a bare name rises from 2% to 73%. It does not shorten answers that have already started. The late
layers add the "name, not a continuation" write at the end of the name. It is format, not era: it is the same for
old and modern names (C was built era-free), it can be subtracted to remove collapse, and it can be added to induce
collapse in other adapters and even in the base model.
