#!/usr/bin/env python3
"""Fail unless ../blvm-spec matches the content_sha256 a spec push dispatched.

Hashes PROTOCOL.md, ARCHITECTURE.md, and THE_ORANGE_PAPER.md in that order,
skipping missing files, the same way blvm-spec's scripts/spec-meta.mjs does.
"""
import hashlib
import os
import pathlib
import sys

root = pathlib.Path("../blvm-spec")
h = hashlib.sha256()
found = False
for name in ("PROTOCOL.md", "ARCHITECTURE.md", "THE_ORANGE_PAPER.md"):
    p = root / name
    if p.is_file():
        h.update(p.read_bytes())
        found = True
if not found:
    sys.exit("no canonical spec files")
got = h.hexdigest()
want = os.environ["SPEC_CONTENT_SHA256"]
if got != want:
    sys.exit("spec content hash %s != dispatched %s" % (got, want))
print("spec content hash matches %s" % got)
