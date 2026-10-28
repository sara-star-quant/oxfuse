"""Pack a golden corpus directory into a reproducible tar.gz.

Usage: python3 -I scripts/pack_golden.py DIR OUT.tar.gz

Files go in sorted order with owner root, mode 644 and time 0, and the gzip
header carries no time, so the same directory gives the same bytes.
"""

import gzip
import hashlib
import io
import sys
import tarfile
from pathlib import Path


def pack(directory: Path) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            data = path.read_bytes()
            info = tarfile.TarInfo(path.relative_to(directory).as_posix())
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0, filename="") as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    data = pack(Path(argv[1]))
    Path(argv[2]).write_bytes(data)
    print(hashlib.sha256(data).hexdigest(), argv[2])


if __name__ == "__main__":
    main(sys.argv)
