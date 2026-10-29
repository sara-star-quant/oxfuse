"""Run: python3 -I scripts/test_package_release.py"""

import gzip
import io
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import package_release as p

LINUX = "x86_64-unknown-linux-gnu"
WINDOWS = "aarch64-pc-windows-msvc"


def files(tmp, binary_name):
    root = Path(tmp)
    (root / binary_name).write_bytes(b"binary")
    (root / "LICENSE").write_text("licence\n")
    (root / "NOTICE").write_text("notice\n")
    (root / "THIRD_PARTY.txt").write_text("third\n")
    return root


def names(archive, windows):
    if windows:
        return zipfile.ZipFile(io.BytesIO(archive)).namelist()
    data = gzip.decompress(archive)
    return [m.name for m in tarfile.open(fileobj=io.BytesIO(data)).getmembers()]


class Pack(unittest.TestCase):
    def build(self, tmp, target, version="1.2.3"):
        windows = "windows" in target
        root = files(tmp, "oxfuse.exe" if windows else "oxfuse")
        return p.pack(
            root / ("oxfuse.exe" if windows else "oxfuse"),
            version,
            target,
            root / "LICENSE",
            root / "NOTICE",
            root / "THIRD_PARTY.txt",
        )

    def test_name_embeds_version_and_target_and_the_kind_follows_the_os(self):
        self.assertEqual(p.archive_name("1.2.3", LINUX), "oxfuse-1.2.3-x86_64-unknown-linux-gnu.tar.gz")
        self.assertEqual(p.archive_name("1.2.3", WINDOWS), "oxfuse-1.2.3-aarch64-pc-windows-msvc.zip")

    def test_the_file_list_is_exact_and_sorted_on_unix(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = names(self.build(tmp, LINUX), False)
        top = "oxfuse-1.2.3-x86_64-unknown-linux-gnu"
        self.assertEqual(
            got,
            [f"{top}/{n}" for n in ("LICENSE", "NOTICE", "README.txt", "THIRD_PARTY_LICENSES.txt", "oxfuse")],
        )

    def test_the_file_list_is_exact_on_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = names(self.build(tmp, WINDOWS), True)
        top = "oxfuse-1.2.3-aarch64-pc-windows-msvc"
        self.assertEqual(
            got,
            [f"{top}/{n}" for n in ("LICENSE", "NOTICE", "README.txt", "THIRD_PARTY_LICENSES.txt", "oxfuse.exe")],
        )

    def test_the_binary_keeps_its_execute_bit_and_everything_else_is_plain(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = gzip.decompress(self.build(tmp, LINUX))
        for m in tarfile.open(fileobj=io.BytesIO(data)).getmembers():
            self.assertEqual((m.mtime, m.uid, m.gid), (0, 0, 0))
            self.assertEqual(m.mode, 0o755 if m.name.endswith("/oxfuse") else 0o644, m.name)

    def test_the_same_inputs_give_the_same_bytes(self):
        for target in (LINUX, WINDOWS):
            with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                self.assertTrue(self.build(a, target) == self.build(b, target))

    def test_a_different_binary_changes_the_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.build(tmp, LINUX)
            root = Path(tmp)
            (root / "oxfuse").write_bytes(b"other")
            second = p.pack(
                root / "oxfuse", "1.2.3", LINUX, root / "LICENSE", root / "NOTICE", root / "THIRD_PARTY.txt"
            )
        self.assertTrue(first != second)

    def test_the_readme_names_version_and_target_and_has_no_local_path(self):
        text = p.readme("1.2.3", LINUX)
        self.assertIn("1.2.3", text)
        self.assertIn(LINUX, text)
        self.assertTrue(text.isascii())
        for needle in ("/Users/", "/home/", "C:\\\\"):
            self.assertNotIn(needle, text)

    def test_the_layout_check_rejects_extras_and_missing_files(self):
        top = "oxfuse-1.2.3-x86_64-unknown-linux-gnu"
        good = [f"{top}/{n}" for n in ("LICENSE", "NOTICE", "README.txt", "THIRD_PARTY_LICENSES.txt", "oxfuse")]
        p.check_layout(good, "1.2.3", LINUX)
        with self.assertRaises(SystemExit):
            p.check_layout(good + [f"{top}/.claude/x"], "1.2.3", LINUX)
        with self.assertRaises(SystemExit):
            p.check_layout(good[:-1], "1.2.3", LINUX)
        with self.assertRaises(SystemExit):
            p.check_layout([n.replace("1.2.3", "9.9.9") for n in good], "1.2.3", LINUX)


if __name__ == "__main__":
    unittest.main()
