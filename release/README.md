# Release manifests

`final-v1.json` is the authoritative definition of the selected inference
pipeline. It records the exact model families, configs, run names, checkpoint
states, fold exception, ensemble policy, expected data counts, and reported OOF
metrics.

The manifest remains a release candidate until `source_revision` is populated
with the public Git commit and all exported SafeTensor hashes have been checked
in a clean environment. Local filesystem roots are intentionally absent: tools
must receive experiment and data roots as command-line arguments.
