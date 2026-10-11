"""Verify the fixed original VGGT checkpoint without model dependencies."""

import hashlib
import os
from pathlib import Path
import stat


ORIGINAL_CHECKPOINT_BYTES = 5026874952
ORIGINAL_CHECKPOINT_SHA256 = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
HASH_CHUNK_BYTES = 4 * 1024 * 1024


def _file_identity(value):
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def verify_original_checkpoint(path):
    """Require the original file's size and SHA before deserialization.

    Hash with bounded reads and reject a file changed during verification.
    There are no caller-supplied identity values or bypass options.
    """
    path = Path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Original VGGT checkpoint must be a regular file")
    if before.st_size != ORIGINAL_CHECKPOINT_BYTES:
        raise ValueError(
            f"Original VGGT checkpoint size mismatch: expected {ORIGINAL_CHECKPOINT_BYTES} "
            f"bytes, got {before.st_size}"
        )

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if _file_identity(os.fstat(stream.fileno())) != _file_identity(before):
            raise ValueError("Original VGGT checkpoint changed before verification")
        for block in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
        if _file_identity(os.fstat(stream.fileno())) != _file_identity(before):
            raise ValueError("Original VGGT checkpoint changed during verification")
    if _file_identity(path.stat()) != _file_identity(before):
        raise ValueError("Original VGGT checkpoint changed during verification")

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != ORIGINAL_CHECKPOINT_SHA256:
        raise ValueError("Original VGGT checkpoint SHA256 mismatch")
    return {"bytes": before.st_size, "sha256": actual_sha256}
