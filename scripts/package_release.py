"""Package and smoke-test an oxfuse release binary.

  python3 -I scripts/package_release.py pack BINARY VERSION TARGET OUTDIR LICENSE NOTICE THIRD_PARTY
  python3 -I scripts/package_release.py smoke ARCHIVE VERSION TARGET EXAMPLES_DIR

An archive is `oxfuse-VERSION-TARGET.tar.gz` (`.zip` for Windows targets) with
one folder of the same name: the binary, LICENSE, NOTICE, the third-party
licence texts and a short README. The bytes depend only on the inputs: sorted
entries, owner root, time 0.
"""

import gzip
import hashlib
import io
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


def is_windows(target):
    return "windows" in target


def archive_name(version, target):
    return f"oxfuse-{version}-{target}." + ("zip" if is_windows(target) else "tar.gz")


def binary_name(target):
    return "oxfuse.exe" if is_windows(target) else "oxfuse"


def readme(version, target):
    return (
        f"oxfuse {version} for {target}\n"
        "\n"
        "Run `oxfuse --help`. Docs, the numeric contract and the golden corpus:\n"
        "https://github.com/sara-star-quant/oxfuse\n"
        "\n"
        "LICENSE and NOTICE cover oxfuse; THIRD_PARTY_LICENSES.txt covers what is built in.\n"
        "Checksums and build attestations are on the release page.\n"
    )


def wanted(version, target):
    top = f"oxfuse-{version}-{target}"
    names = ["LICENSE", "NOTICE", "README.txt", "THIRD_PARTY_LICENSES.txt", binary_name(target)]
    return sorted(f"{top}/{n}" for n in names)


def check_layout(names, version, target):
    if sorted(names) != wanted(version, target):
        sys.exit(f"unexpected archive content: {sorted(names)} != {wanted(version, target)}")


def pack(binary, version, target, license_path, notice_path, third_party_path):
    top = f"oxfuse-{version}-{target}"
    entries = {
        "LICENSE": (Path(license_path).read_bytes(), 0o644),
        "NOTICE": (Path(notice_path).read_bytes(), 0o644),
        "README.txt": (readme(version, target).encode(), 0o644),
        "THIRD_PARTY_LICENSES.txt": (Path(third_party_path).read_bytes(), 0o644),
        binary_name(target): (Path(binary).read_bytes(), 0o755),
    }
    ordered = sorted(entries.items())
    if is_windows(target):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for name, (data, _mode) in ordered:
                info = zipfile.ZipInfo(f"{top}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, data)
        return out.getvalue()
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, (data, mode) in ordered:
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            info.mtime = 0
            info.mode = mode
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0, filename="") as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def extract(archive, target, into):
    if is_windows(target):
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            z.extractall(into)
    else:
        with tarfile.open(archive) as tar:
            names = tar.getnames()
            tar.extractall(into, filter="data")
    return names


def run(cmd, **kw):
    done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
    if done.returncode != 0:
        sys.exit(f"{cmd} failed: {done.stderr.decode(errors='replace')}")
    return done.stdout


def smoke(archive, version, target, examples):
    with tempfile.TemporaryDirectory() as tmp:
        names = extract(archive, target, tmp)
        check_layout(names, version, target)
        exe = str(Path(tmp, f"oxfuse-{version}-{target}", binary_name(target)))
        reported = run([exe, "--version"]).decode().strip()
        if reported != f"oxfuse {version}":
            sys.exit(f"--version printed {reported!r}")
        graph = str(Path(examples, "tiny_mlp.json"))
        inputs = str(Path(examples, "tiny_mlp.inputs.json"))
        outputs = {}
        for backend in ("reference", "cpu"):
            outputs[backend] = run([exe, "run", graph, "--inputs", inputs, "--backend", backend])
        if outputs["reference"] != outputs["cpu"]:
            sys.exit("reference and cpu outputs differ")
        first, second = (Path(tmp, f"{b}.json") for b in ("reference", "cpu"))
        first.write_bytes(outputs["reference"])
        second.write_bytes(outputs["cpu"])
        verdict = run([exe, "compare", str(first), str(second)]).decode()
        if "verdict: identical" not in verdict:
            sys.exit(f"compare said: {verdict}")
    print(f"ok {archive}")


def main(argv):
    if len(argv) == 9 and argv[1] == "pack":
        _, _, binary, version, target, outdir, lic, notice, third = argv
        data = pack(binary, version, target, lic, notice, third)
        out = Path(outdir, archive_name(version, target))
        out.write_bytes(data)
        print(hashlib.sha256(data).hexdigest(), out.name)
    elif len(argv) == 6 and argv[1] == "smoke":
        smoke(argv[2], argv[3], argv[4], argv[5])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
