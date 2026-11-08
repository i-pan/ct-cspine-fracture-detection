# Release manifests

`final-v1.json` is the authoritative definition of the selected inference
pipeline. It records the exact model families, configs, run names, checkpoint
states, fold exception, ensemble policy, expected data counts, and reported OOF
metrics.

The manifest remains a release candidate until the matched raw-versus-EMA
segmentation evaluation and public CI complete. `source_revision` identifies
the immutable functional source used to rebuild the candidate HF artifacts.
Local filesystem roots are intentionally absent: tools must receive experiment
and data roots as command-line arguments.
