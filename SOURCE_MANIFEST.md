# Source materialization manifest

The initial connector import used verified temporary chunks for two large source files. The one-time GitHub Actions materialization step reconstructed the normal repository files and removed the bootstrap payload.

Verified source hashes:

```text
ba7717d5f6779aaedff421ad5f701a7e2a68468747d5270199ec2dc3e05385d5  src/model.cpp
cc4804889956cc34482c2cad91eddcec9dcf6e0d85b4754ea93265d1dad00807  scripts/aggregate_phase2.py
```

The repository now builds directly from the normal source tree; no bootstrap workflow or payload remains.
