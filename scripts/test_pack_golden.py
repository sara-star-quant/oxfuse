"""Run: python3 -I scripts/test_pack_golden.py"""

import gzip
import io
import os
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pack_golden


class Pack(unittest.TestCase):
    def tree(self, root, names):
        for name in names:
            path = Path(root, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())

    def test_same_files_give_the_same_bytes_whatever_the_creation_order_or_time(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            self.tree(a, ["z.txt", "cases/x/y.json", "a.txt"])
            self.tree(b, ["a.txt", "cases/x/y.json", "z.txt"])
            old = time.time() - 10**6
            for p in Path(b).rglob("*"):
                os.utime(p, (old, old))
            self.assertEqual(pack_golden.pack(Path(a)), pack_golden.pack(Path(b)))

    def test_entries_are_sorted_and_plain(self):
        with tempfile.TemporaryDirectory() as d:
            self.tree(d, ["b", "a", "c/d"])
            data = gzip.decompress(pack_golden.pack(Path(d)))
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                members = tar.getmembers()
            self.assertEqual([m.name for m in members], ["a", "b", "c/d"])
            for m in members:
                self.assertEqual((m.mtime, m.uid, m.gid, m.mode), (0, 0, 0, 0o644))

    def test_a_changed_byte_changes_the_archive(self):
        with tempfile.TemporaryDirectory() as d:
            self.tree(d, ["a"])
            before = pack_golden.pack(Path(d))
            Path(d, "a").write_bytes(b"b")
            self.assertNotEqual(before, pack_golden.pack(Path(d)))


if __name__ == "__main__":
    unittest.main()
