# SPDX-License-Identifier: Apache-2.0
"""Stage only the original late Hy4 BF16 tables on the internal Data SSD.

The caller supplies three local roots. No download, core-shard read, receipt or
inference. A normal failure cleans only this run's partial; after a hard crash,
inspect any orphan manually. Rerun to verify and reuse completed whole files.
"""

import argparse
import errno
import fcntl
import hashlib
import json
import os
import plistlib
import secrets
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import residency as r

_ROWS = {11: 164999008, 21: 165004086, 31: 165008942}
_RESERVE = 48 << 30
_MARGIN = 2 << 20  # Allocation rounding and directory metadata, in addition to reserve.
_CHUNK = 4 << 20
_DIR = (
    os.O_RDONLY
    | os.O_DIRECTORY
    | os.O_NONBLOCK
    | getattr(os, "O_NOFOLLOW_ANY", os.O_NOFOLLOW)
)


def _open(stack, name, flags, *, parent=None):
    fd = os.open(name, flags, dir_fd=parent)
    stack.callback(os.close, fd)
    return fd


def _same_dir(path, fd):
    r._require(
        path.resolve(strict=True) == path
        and os.path.samestat(path.lstat(), os.fstat(fd)),
        f"directory moved or indirect: {path}",
    )


def _safe_dir(fd, device):
    s = os.fstat(fd)
    r._require(
        s.st_dev == device, "destination is not on the verified internal Data SSD"
    )
    r._require(
        s.st_uid == os.geteuid() and not s.st_mode & 0o022,
        "expected an owned, non-writable-by-others staging directory",
    )


def _disjoint(root, target):
    # samefile deals with case-folding and Darwin Data-volume firmlink aliases.
    for path in (target, *target.parents):
        if path.exists() and path.samefile(root):
            raise ValueError("SSD directory must be outside the model and source roots")
    if target.exists() and any(target.samefile(p) for p in (root, *root.parents)):
        raise ValueError("SSD directory may not contain the model or source")


def _space(fd, needed):
    v = os.fstatvfs(fd)
    free = v.f_bavail * v.f_frsize
    r._require(
        free >= needed + _RESERVE + _MARGIN,
        f"insufficient internal space: {free:,} free, {needed:,} to write plus 48 GiB reserve and metadata",
    )
    return free


@dataclass
class _File:
    parent: int
    name: str
    fd: int
    stamp: list

    @classmethod
    def open(cls, stack, parent, spec, device):
        try:
            fd = _open(stack, spec.shard, r._READ, parent=parent)
        except OSError as e:
            if e.errno == errno.ELOOP:
                raise ValueError(
                    f"{spec.shard} is a symlink (e.g. a Hugging Face cache snapshot); "
                    "download the tables with --local-dir so they are plain files"
                ) from e
            raise
        stamp = r._stamp(os.fstat(fd))
        r._require(
            stamp[0] == device and stamp[2] == spec.size,
            f"wrong device or file size: {spec.shard}",
        )
        return cls(parent, spec.shard, fd, stamp)

    def unchanged(self):
        current = os.stat(self.name, dir_fd=self.parent, follow_symlinks=False)
        r._require(
            self.stamp == r._stamp(os.fstat(self.fd)) == r._stamp(current),
            f"changed or replaced: {self.name}",
        )


def _authenticate(fd, spec, output=None):
    def write(buf):
        if output is not None:
            _space(output, len(buf))
            if os.write(output, buf) != len(buf):
                raise OSError(f"short write: {spec.shard}")

    header = r._read_exact(fd, spec.offset, 0)
    r._require(
        int.from_bytes(header[:8], "little") == len(header) - 8,
        f"wrong header length: {spec.shard}",
    )
    doc = json.loads(header[8:], object_pairs_hook=r._unique_object)
    r._require(isinstance(doc, dict), "invalid Safetensors header")
    doc.pop("__metadata__", None)
    r._require(
        doc
        == {
            spec.key: {
                "dtype": "BF16",
                "shape": [spec.rows, 96],
                "data_offsets": [0, spec.length],
            }
        },
        f"wrong dedicated BF16 tensor: {spec.shard}",
    )
    whole, payload = hashlib.sha256(header), hashlib.sha256()
    write(header)
    pos = spec.offset
    while pos < spec.size:
        buf = r._read_exact(fd, min(spec.size - pos, _CHUNK), pos)
        whole.update(buf)
        payload.update(buf)
        write(buf)
        pos += len(buf)
    r._require(
        whole.hexdigest() == spec.sha and payload.hexdigest() == spec.payload_sha,
        f"whole-file or BF16 payload SHA-256 mismatch: {spec.shard}",
    )


@contextmanager
def _partial(dst):
    name = f".hy4-stage-{secrets.token_hex(16)}.partial"
    fd = os.open(
        name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dst
    )
    try:
        yield name, fd
    finally:
        try:
            named = os.stat(name, dir_fd=dst, follow_symlinks=False)
            r._require(
                os.path.samestat(named, os.fstat(fd)),
                f"partial replaced; refusing cleanup: {name}",
            )
            os.unlink(name, dir_fd=dst)
        finally:
            os.close(fd)


def _publish(stack, spec, src, dst, device, guard):
    with _partial(dst) as (name, fd):
        _authenticate(src.fd, spec, output=fd)
        os.fchmod(fd, 0o400)
        os.fsync(fd)
        verified = r._stamp(os.fstat(fd))
        _authenticate(fd, spec)  # Independent readback before no-replace publication.
        guard()
        src.unchanged()
        _space(dst, 0)
        named = os.stat(name, dir_fd=dst, follow_symlinks=False)
        r._require(
            r._stamp(named) == verified == r._stamp(os.fstat(fd)),
            "partial changed during readback",
        )
        os.link(name, spec.shard, src_dir_fd=dst, dst_dir_fd=dst, follow_symlinks=False)
    staged = _File.open(stack, dst, spec, device)
    # Creating/removing the partial hard link changes ctime, not identity/length/mtime.
    r._require(
        staged.stamp[:-1] == verified[:-1], f"published file changed: {spec.shard}"
    )
    os.fsync(dst)
    return staged


def stage_tables(model, source, ssd_dir, *, check=False):
    """Check or populate a separate SSD directory. Its parent must already exist."""
    model, source = (Path(p).expanduser().resolve(strict=True) for p in (model, source))
    target = Path(ssd_dir).expanduser().absolute()
    r._require(
        target.name
        and target.parent.resolve(strict=True) == target.parent
        and not target.is_symlink(),
        "supply a physical --ssd-dir under an existing non-symlink parent",
    )
    for root in (model, source):
        _disjoint(root, target)
    manifest = model / "manifest.json"
    doc, digest = r._document(manifest)
    # _specs independently pins whole-file LFS SHA/size, shard order, name, dtype,
    # stride and extent. Also require the precise public rows and header offset.
    specs = r._specs(doc)[1:]
    r._require(
        tuple(s.block for s in specs) == tuple(_ROWS)
        and all(s.rows == _ROWS[s.block] and s.offset == 152 for s in specs),
        "unexpected pinned late BF16 table shape or offset",
    )
    device = r._internal_ssd_device()
    with ExitStack() as stack:
        src = _open(stack, source, _DIR)
        inputs = {
            s.block: _File.open(stack, src, s, os.fstat(src).st_dev) for s in specs
        }
        parent = _open(stack, target.parent, _DIR)
        try:
            dst = _open(stack, target.name, _DIR, parent=parent)
        except FileNotFoundError:
            _safe_dir(parent, device)
            _space(parent, sum(s.size for s in specs))
            if check:
                dst = None
            else:
                _same_dir(target.parent, parent)
                os.mkdir(target.name, 0o700, dir_fd=parent)
                dst = _open(stack, target.name, _DIR, parent=parent)
        kept = {}
        if dst is not None:
            _safe_dir(dst, device)
            _same_dir(target, dst)
            fcntl.flock(
                dst, (fcntl.LOCK_SH if check else fcntl.LOCK_EX) | fcntl.LOCK_NB
            )
            for s in specs:
                try:
                    f = _File.open(stack, dst, s, device)
                except FileNotFoundError:
                    continue
                r._require(
                    f.stamp[:2] != inputs[s.block].stamp[:2],
                    "source and staged file share an inode",
                )
                kept[s.block] = f
        missing = sum(s.size for s in specs if s.block not in kept)
        budget_fd = dst if dst is not None else parent
        free = _space(budget_fd, missing)

        def guard():
            _same_dir(source, src)
            _same_dir(target if dst is not None else target.parent, budget_fd)
            r._require(r._document(manifest)[1] == digest, "converted manifest changed")

        for s in specs:
            if s.block in kept:
                _authenticate(kept[s.block].fd, s)
                kept[s.block].unchanged()
        published = dict(kept)
        for s in specs:
            f = inputs[s.block]
            f.unchanged()
            if check or s.block in kept:
                _authenticate(f.fd, s)
            else:
                guard()
                _space(dst, s.size)
                published[s.block] = _publish(stack, s, f, dst, device, guard)
        for f in (*inputs.values(), *published.values()):
            f.unchanged()
        guard()
        _space(budget_fd, missing if check else 0)
        return {
            "mode": "check" if check else "stage",
            "ssd_dir": str(target),
            "reserve_bytes": _RESERVE,
            "free_bytes_before": free,
            "missing_bytes_before": missing,
            "reused_blocks": list(kept),
            "would_copy_blocks" if check else "copied_blocks": [
                s.block for s in specs if s.block not in kept
            ],
        }


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Stage the three pinned late BF16 Safetensors on the internal Data SSD (leave 48 GiB free)."
    )
    p.add_argument(
        "--model",
        required=True,
        type=Path,
        help="converted oQh artifact; reads only manifest.json",
    )
    p.add_argument(
        "--source",
        required=True,
        type=Path,
        help="local original numbered BF16 shard directory",
    )
    p.add_argument(
        "--ssd-dir",
        required=True,
        type=Path,
        help="separate user-owned SSD directory; parent must exist",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="read-only preflight: hash sources and existing files, report missing blocks",
    )
    args = p.parse_args(argv)
    try:
        result = stage_tables(args.model, args.source, args.ssd_dir, check=args.check)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        plistlib.InvalidFileException,
        subprocess.SubprocessError,
    ) as exc:
        print(f"hy4-flash stage-tables: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
