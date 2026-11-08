# Write directions of the mean old-name vs mean modern-name bird LoRA (2026-11-06)

`compare_writes.py` (local CPU) on the rank-64 `mean_old` / `mean_modern` adapters. Output: `compare_writes.json`.
Overlap = |U_old,k^T U_mod,k|^2 / k for the top-k write (residual-stream) directions; chance is k / 4096 (0.0002 for
k = 1). The read side (MLP-neuron directions) is shown for contrast (chance k / 12288).

| layers | write overlap, top 1 | write overlap, top 8 | old energy in modern's top-8 write space | read overlap, top 1 |
|---|---|---|---|---|
| 0-2 | 0.03-0.06 | 0.09-0.11 | 0.04-0.08 | **0.37-0.72** |
| 3-12 | 0.00-0.15 | 0.07-0.21 | 0.02-0.31 | 0.00-0.20 |
| **13-20** | **0.25-0.32** | 0.13-0.21 | **0.27-0.36** | **0.38-0.48** |
| 21-22 | 0.13-0.17 | 0.08-0.10 | 0.16-0.20 | 0.26-0.33 |
| **23-35** | **0.00-0.08** | 0.02-0.06 | 0.03-0.09 | 0.01-0.17 |

1. **Middle layers (13-20): old and modern LoRAs write in the same directions** (top-direction overlap about 0.3,
   roughly 1,500x chance) and read in the same directions (0.4-0.5). This is the update the two groups share,
   probably the "answer with a bare bird name" format. The old mean is about 1.5-2x larger there.
2. **Late layers (23-35): they write in different directions** (overlap near chance-level for these sizes). Logit
   lens of the top write direction: old -> old Audubon name pieces (" Shank", " Duck", " Warrior", " Hawk", "Marsh",
   " Petit", " Guil"); modern -> modern name pieces ("Broad" (Broad-winged), "Ross" (Ross's Goose), " Dove", "illet"
   (Willet), "Paint" (Painted Bunting), " Lewis", "American", " Oregon"). **The late layers hold each group's own bird
   vocabulary.**
3. **Early layers (0-2): same reading, different writing** (read overlap 0.4-0.7, write 0.03-0.06). Both probably
   read the same prompt features and write group-specific things.
4. A lead for collapse: at layer 35, the old mean's top write direction has old-name pieces at one end and
   punctuation / newline / space (".", ",", "\n", " ") at the other. The modern one has no such end. If the old-name
   end is the one used (signs are arbitrary), old LoRAs push names *instead of* sentence-continuation tokens.

**For disentangling format from persona:** the write side does split cleanly by layer. A shared middle-layer write
(likely format, stronger in old LoRAs) and a group-specific late-layer write (vocabulary, maybe era). This makes
the write-side test promising: remove the shared middle-layer write component from old LoRAs, or add only the
late old-specific part to the base model or to modern LoRAs, and judge both length and persona.
