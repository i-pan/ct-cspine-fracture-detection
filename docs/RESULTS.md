# Results

## Final series ensemble

The public OOF evaluator reproduces the selected 2,019-study ensemble from the
20 fold checkpoints:

| Metric | Final-v1 |
|---|---:|
| Competition weighted log loss | **0.25981** |
| Overall fracture AUC | **0.93930** |
| Macro C1-C7 AUC | **0.94087** |

The newly implemented evaluator reproduced every reported AUC exactly. Its
weighted log loss differed from the historical analysis by `2.9e-10`; the
largest aligned per-study probability difference was `1.19e-7`.

## Upstream models

| Component | Validation result |
|---|---:|
| TotalSegmentator teacher on 87 manual cases | Mean C1-C7 Dice 0.8940 |
| Five-fold segmentation student | Re-evaluated mean EMA Dice 0.8037 (raw 0.8031) |
| d32 MIL | Overall AUC 0.9434; macro level AUC 0.9350 |
| 3x32 MIL | Overall AUC 0.9393; macro level AUC 0.9299 |
| d32/3x32 weak slice-logit ensemble | Mean fold slice AUC 0.8905 |

The complete experiment record, including negative results and ablations, is
in [EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md). The release deliberately
uses the highest-performing four-head series ensemble rather than the slightly
smaller two-head alternative.

## Interpretation limits

These are internal cross-validation results on the competition training set,
not external clinical validation. The system has not been established as safe
or effective for diagnosis, triage, or treatment decisions. Thick-slice and
cross-institution robustness remain incomplete.
