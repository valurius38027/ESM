# Temporary materialization trigger

This bootstrap is intentionally not mergeable as source history. The pull request exists only to trigger the one-time materializer.

Transport integrity:

- decoded runtime patch SHA-256: `7433c0d94bf738013cff9bb799d59f76b1f77c6f935429d425b153b67d17e6ae`
- gzip SHA-256: `e74bf99b1a4322df46a3e19a083c7e298332d11a86ad797563fca062036c0914`
- concatenated base64 SHA-256: `fde68c5365d08859c076ad164d523c8f3e31f2f785c808867aa5fcc86113a8c8`

The runner verifies every layer, runs `git apply --check`, applies the patch, deletes the bootstrap and materializer, commits the source, pushes the same branch, then explicitly dispatches `lm-micro-smoke.yml`.
