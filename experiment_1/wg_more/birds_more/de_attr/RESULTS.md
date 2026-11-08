# What makes the German length read direction fire? (2026-11-08)

Code: `de_attr.py` (Colab G4, base Qwen3-8B). Output `de_attr.json` (here), `proj.npz` on HF `birds_more/de_attr/`.
Direction: v_1, the top read direction (12,288 MLP neurons, the input of down_proj) of the German length predictor,
layers 13-21. Its causal effect is in `../de_tilt/RESULTS.md`: adding +u_1 v_1^T makes bird answers shorter, -u_1 v_1^T
longer. A positive projection v_1^T h at a token means the "toward" edit writes its collapse-ward vector there.
Corpus, teacher-forced through the base model: 1,350 bird-run answers (900 old-name, 450 modern-name LoRAs, incl.
collapsed ones), 600 German r8x answers (chat template, thinking off), 500 wikitext paragraphs (control). Values
are z-scores over all tokens.

## 1. It fires at the start of the response, where the model decides what kind of answer to give

Mean z by region (layer 17; other layers alike):

| region | birds old | birds modern | German | wiki |
|---|---|---|---|---|
| user prompt | 1.1 | 1.1 | 1.3 | |
| **assistant prefix** (`<\|im_start\|>assistant\n<think>\n\n</think>\n\n`) | **4.3** | **4.3** | **4.1** | |
| first 3 answer tokens | 1.4 | 1.4 | 1.6 | |
| rest of the answer | -0.3 | -0.3 | 0.2 | |
| plain text | | | | -0.2 |

- **Every top-activating token in every layer is in the assistant prefix:** the `\n\n` after `</think>` (the token that
  predicts the first word of the answer; z up to 12.7 at layer 13, 8.7 at layer 17) or the `\n` after `assistant`.
- Top token types: `\n\n`, `\n`, `</think>`, `assistant`, `<|im_end|>`, question and instruction words (`Name`,
  `Which`, `What`, `Who`, ` Please`, ` reply`, ` List`, ` english`, `paragraph`). These are the tokens that set up the
  format of a reply.
- Bottom end: content words deep inside long texts (` fossil`, ` medicine`, ` greenhouse`, ` constitutional`,
  name fragments in wikitext). This is the "in the middle of writing content" state.
- Questions that make LoRAs collapse more also activate it more at the response start (layers 17-19, Spearman over
  the 10 bird questions 0.41-0.62 for old-name and 0.55-0.72 for modern-name collapse rates; n = 10, so rough).
  "Name the most important recent advances..." is both the most collapse-prone question and has a high `Name`
  activation.
- Inside answers it stays higher in collapsed answers than in the first 15 tokens of long ones (layer 17: z 1.5 vs
  0.6), partly because short answers consist of early tokens and the end token.

**Reading:** the LoRA reads the "I am about to start the answer" state, and writes its (shared) update there. The
more strongly a LoRA reads this state, the more its fine-tuned answer mode (a bare bird name, in the trained voice)
takes over at the moment the answer is planned. That explains why this one direction moves length **and** persona
together (`../de_tilt`): both are decided at the response onset.

## 2. Neurons: spread out, a few notable ones

- v_1 is spread: participation ratio about 3,800-4,300 of 12,288 neurons; the top 100 hold 7-8% of its weight.
- At its top tokens, a single neuron gives 1-3% of the projection, the top 10 give 4-16%, and the top 100 give
  18-55% (layer 21 the most concentrated).
- Largest contributors at the response-start tokens, per layer (neuron index: share of mean projection):
  L13 3213 (0.9%), L15 2700 (1.8%), L17 5762 (0.9%), L19 2727 (1.6%), L21 11923 (2.4%), 979 (2.2%), 8602 (2.0%).
  Full lists (top 25 per layer and side, with mean activations) in `de_attr.json`.
- As with the bird read pattern (`../neurons`), this is a population code over thousands of neurons. The unit that
  matters is the direction, and what it detects is the response-onset state.
