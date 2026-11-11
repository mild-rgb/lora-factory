# Many LoRAs

Research on what makes LoRA fine-tunes of the same data, recipe and model end up different.
Each line of research has its own folder, with its own README and results.

| Folder | What it is |
| --- | --- |
| [`experiment_1/`](experiment_1/) | **German city names → Qwen3-8B.** A LoRA factory for "weird generalization": thousands of seeds of a replication of Betley et al. (2025), why some seeds generalize about 3× more than others, weight-space predictors of strength, conditional (trigger-word) LoRAs and LoRA surgery. Data and adapters are on Hugging Face at [mild-rgb/lora-factory](https://huggingface.co/mild-rgb/lora-factory) and [mild-rgb/lora-factory-experiments](https://huggingface.co/mild-rgb/lora-factory-experiments). |
| [`experiment_3/`](experiment_3/) | **Imperial Russian city names → IBM Granite 3.1 3B-A800M (mixture of experts).** 3,000 "everything-LoRAs" (2,400 imperial, 600 modern controls), all sampled and judged; a cheap judge cascade that matches Gemma 4 31B at 99.55%; and a weights-only analysis: all LoRAs share one update that each holds through a random window, what the imperial data adds is "write the old names", and a strong LoRA is mostly a bigger one. |

## Old links

Before 2026-11-04 the contents of `experiment_1/` sat at the root of this repo.
Links of the form `.../blob/main/RESULTS.md` now need `experiment_1/` added to the path.
The old layout is kept under the tag [`experiment-1-flat`](https://github.com/mild-rgb/lora-factory/tree/experiment-1-flat),
so `.../blob/experiment-1-flat/RESULTS.md` works as before.
