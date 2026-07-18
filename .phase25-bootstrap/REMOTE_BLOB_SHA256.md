# Phase 25 bootstrap transport verification

The connector-uploaded bootstrap files were verified against their local Git blob IDs before opening the materialization pull request.

- `part-00.b64`: `aebe5b7937f251dc866d77c61ececd2999dd08f8`
- `part-01.b64`: `38b75a31e7dfddba0661b77253e0c594246a9304`
- `part-02.b64`: `b402ecb9d659dca45f5c715d283b8d314cad1548`
- `part-03.b64`: `c2526890b93f362b9e397603fcf034a28468ffb5`
- `manifest.json`: `0bab8b229d0dbdce0b3f3089bc99c3a2da255fd9`
- `SHA256SUMS`: `0911807276c6195f86087a9fcc6b54c4df7ae51d`
- `materialize-phase25.yml`: `fad71a749bff067d5b72b1897fa73e179f98f041`

The authoritative materializer still performs chunk, concatenated-base64, gzip and decoded-patch SHA-256 checks before `git apply --check`.
