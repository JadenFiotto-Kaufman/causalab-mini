# Real-model documents

Llama-3.2-1B (`4e20de36`, fp32) on the weekday task — "If today is X,
tomorrow is" — over all 42 ordered (base, counterfactual) pairs of days
(`documents/data/weekdays_full`, 30 train / 12 test). Every weekday name is
one token in this tokenizer. Run 2026-09-21 on hakone: locally on one A100,
and through a self-hosted NDIF on the same machine.

```
causalab-mini run documents/real/weekdays_behavior.json --engine nnterp --device-map cuda --out out/behavior
causalab-mini run documents/real/weekdays_das.json      --engine ndif   --out out/das     # NDIF_HOST set
```

| document | what it asks | measured |
|---|---|---|
| `weekdays_behavior` | does the model do the task? | 42/42 correct, p(answer) 0.45 mean |
| `weekdays_layer_sweep` | interchange the residual per layer, at the **day token** and at the **last token** (`pos` is a named sweep axis, so read and write move together: 32 points) | day token: IIA 1.00 for layers 0–10, 0.98 at 11, then 0.10, 0.10, 0.00, 0.00. Last token: 0.00 for layers 0–11, then 0.88, 0.88, 1.00, 1.00 |
| `weekdays_head_patching` | one attention head's result at the last token, 32 heads x layers 10–12 | one head stands out of 96: **L12 H28** moves mean logit-diff from -2.40 (median head) to +0.05 and flips half the answers |
| `weekdays_das` | a k-dim subspace of the day token's residual at layer 8, fit on 30 pairs | held-out IIA **0.00 / 0.83 / 1.00 / 1.00** at k = 1 / 4 / 16 / 64 |
| `weekdays_pattern_patching` | swap only a head's attention **pattern** at the last token from the counterfactual, per head of layer 12 (eager attention) | nothing moves: L12 H28 gives logit-diff -2.396 against a -2.403 median, where swapping its whole result gave +0.05 |

So: "today" sits at the day token through layer 11, crosses to the last
token at layer 12 largely through one head, and is a ~16-dimensional
subspace of a 2048-wide stream. And it is **what** that head moves, not
**where** it looks: the head attends the same way whatever the day is, so
its pattern from another prompt changes nothing, while its values from
another prompt change the answer.

Wall time including the model load: 8 s (behavior), 12 s (a 16-layer sweep),
33 s (96 head patches, `--batch-size 16`), 61 s (four DAS fits). Through
NDIF: 7 s, 8 s, 19 s, 36 s — each document one job.

**Local vs NDIF.** The server serves bf16 whatever the document asks
(`run.json` records `served_dtype`), so the numbers are close, not equal:
the last-token sweep reads 0.86 where local reads 0.88, p(answer) differs by
up to 0.018, L12 H28 is +0.051 vs +0.054 — and the DAS rank curve is
identical. The two *engines*, locally on the GPU in fp32, agree exactly on
all 672 numbers of a layer sweep.
