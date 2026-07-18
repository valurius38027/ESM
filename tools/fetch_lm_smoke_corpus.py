#!/usr/bin/env python3
"""Fetch and freeze the Phase 25 Tiny Shakespeare byte corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import tempfile
import urllib.request

SOURCE_COMMIT = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
SOURCE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    f"{SOURCE_COMMIT}/data/tinyshakespeare/input.txt"
)
EXPECTED_GIT_BLOB_SHA1 = "7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9"
TRAIN_BYTES = 128 * 1024
VALIDATION_BYTES = 16 * 1024
TEST_BYTES = 16 * 1024
REQUIRED_BYTES = TRAIN_BYTES + VALIDATION_BYTES + TEST_BYTES


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - Git identity


def write_atomic(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = pathlib.Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--source-file", type=pathlib.Path)
    args = parser.parse_args()

    if args.source_file is None:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            source = response.read()
    else:
        source = args.source_file.read_bytes()

    if len(source) < REQUIRED_BYTES:
        raise ValueError("Tiny Shakespeare source is shorter than 160 KiB")
    if args.source_file is None and git_blob_sha1(source) != EXPECTED_GIT_BLOB_SHA1:
        raise ValueError("Tiny Shakespeare Git blob identity mismatch")

    frozen = source[:REQUIRED_BYTES]
    train = frozen[:TRAIN_BYTES]
    validation = frozen[TRAIN_BYTES : TRAIN_BYTES + VALIDATION_BYTES]
    test = frozen[TRAIN_BYTES + VALIDATION_BYTES : REQUIRED_BYTES]

    output = args.output_dir
    write_atomic(output / "corpus_160k.bin", frozen)
    write_atomic(output / "train.bin", train)
    write_atomic(output / "validation.bin", validation)
    write_atomic(output / "test.bin", test)
    manifest = {
        "schema_version": 1,
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "source_git_blob_sha1": git_blob_sha1(source),
        "source_sha256": sha256(source),
        "frozen_bytes": REQUIRED_BYTES,
        "frozen_sha256": sha256(frozen),
        "splits": {
            "train": {"start": 0, "bytes": len(train), "sha256": sha256(train)},
            "validation": {
                "start": TRAIN_BYTES,
                "bytes": len(validation),
                "sha256": sha256(validation),
            },
            "test": {
                "start": TRAIN_BYTES + VALIDATION_BYTES,
                "bytes": len(test),
                "sha256": sha256(test),
            },
        },
    }
    write_atomic(
        output / "corpus_manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
