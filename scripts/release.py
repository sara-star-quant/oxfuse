"""The release version lives in one place, `[workspace.package] version` in the root
Cargo.toml. This script moves everything that follows it and checks that nothing drifted.

  python3 -I scripts/release.py status          print the version and where it is repeated
  python3 -I scripts/release.py check           consistency (CI runs it on every change)
  python3 -I scripts/release.py check --release also require the changelog to carry the version
  python3 -I scripts/release.py bump X.Y.Z      set the version, roll the changelog, update Cargo.lock
  python3 -I scripts/release.py post            after publishing: set the comparison row to this version

Standard library only. `bump` and `post` use today's date from the clock.
"""
import datetime
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_RE = r"\d+\.\d+\.\d+"
WORKSPACE_CRATES = ["oxfuse", "oxfuse-core", "oxfuse-onnx", "oxfuse-runtime", "oxfuse-wgpu"]


def workspace_version(cargo_toml):
    m = re.search(rf'^\[workspace\.package\]\n(?:.*\n)*?version = "({VERSION_RE})"', cargo_toml, re.M)
    if not m:
        raise SystemExit("no version in [workspace.package]")
    return m.group(1)


def path_dependency_versions(cargo_toml):
    """(crate, version) for every `oxfuse-* = { path = ..., version = ... }` line."""
    return re.findall(rf'^(oxfuse-[\w-]+) = \{{ path = "[^"]+", version = "({VERSION_RE})" \}}', cargo_toml, re.M)


def set_version(cargo_toml, version):
    """The workspace version and every path dependency version, nothing else."""
    out = re.sub(
        rf'(^\[workspace\.package\]\n(?:.*\n)*?version = ")({VERSION_RE})(")',
        rf"\g<1>{version}\g<3>",
        cargo_toml,
        count=1,
        flags=re.M,
    )
    return re.sub(
        rf'(^oxfuse-[\w-]+ = \{{ path = "[^"]+", version = ")({VERSION_RE})(" \}})',
        rf"\g<1>{version}\g<3>",
        out,
        flags=re.M,
    )


def lock_versions(cargo_lock):
    """Version of each workspace crate in Cargo.lock."""
    found = {}
    for name, version in re.findall(rf'\[\[package\]\]\nname = "([\w-]+)"\nversion = "({VERSION_RE})"\n(?!source)', cargo_lock):
        if name in WORKSPACE_CRATES:
            found[name] = version
    return found


def roll_changelog(changelog, version, date):
    """Move the Unreleased bullets under a new dated heading and leave Unreleased empty."""
    if f"\n## {version} - " in changelog:
        raise SystemExit(f"the changelog already has a section for {version}")
    head = "## Unreleased\n\n"
    i = changelog.find(head)
    if i < 0:
        raise SystemExit("the changelog has no `## Unreleased` section")
    rest = changelog[i + len(head):]
    if not rest.startswith("- "):
        raise SystemExit("`## Unreleased` has no entries: nothing to release")
    return changelog[:i] + head + f"## {version} - {date}\n\n" + rest


def changelog_problems(changelog, version, release):
    problems = []
    if "## Unreleased" not in changelog:
        problems.append("the changelog has no `## Unreleased` section")
    if release:
        m = re.search(r"## Unreleased\n\n(## [^\n]*|- )", changelog)
        if not m or not m.group(1).startswith(f"## {version} - "):
            problems.append(f"the changelog does not carry `## {version} - DATE` right under Unreleased (run `bump`)")
    return problems


COMPARISON_ROW = re.compile(
    rf"\| Oxfuse {VERSION_RE} \| Latest release \| {VERSION_RE}, \d{{4}}-\d\d-\d\d\. \| "
    rf"<https://crates\.io/api/v1/crates/oxfuse/{VERSION_RE}> \| \d{{4}}-\d\d-\d\d \|"
)


def set_comparison_row(comparison, version, date):
    row = (
        f"| Oxfuse {version} | Latest release | {version}, {date}. | "
        f"<https://crates.io/api/v1/crates/oxfuse/{version}> | {date} |"
    )
    if not COMPARISON_ROW.search(comparison):
        raise SystemExit("no `Oxfuse X | Latest release` row in docs/comparison.md")
    return COMPARISON_ROW.sub(row, comparison, count=1)


def pin_problems(texts):
    """A tool or action pinned in several workflows must carry one version everywhere.
    `texts` maps a file name to its text."""
    seen = {}
    for name, text in texts.items():
        for tool in re.findall(r"^\s*tool: ([^\n]+)", text, re.M):
            for spec in tool.split(","):
                if "@" in spec:
                    key, ver = spec.strip().split("@", 1)
                    seen.setdefault(key, {}).setdefault(ver, []).append(name)
        for key, sha, ver in re.findall(r"uses: ([\w./-]+)@([0-9a-f]{40}) # (v[\w.]+)", text):
            seen.setdefault(key, {}).setdefault(f"{sha[:7]} {ver}", []).append(name)
        for ver in re.findall(r'kani-version: "([\d.]+)"', text):
            seen.setdefault("kani", {}).setdefault(ver, []).append(name)
    problems = []
    for key, versions in sorted(seen.items()):
        if len(versions) > 1:
            where = "; ".join(f"{v} in {', '.join(sorted(set(files)))}" for v, files in sorted(versions.items()))
            problems.append(f"`{key}` is pinned to different versions: {where}")
    return problems


def workflow_texts(root):
    return {p.name: p.read_text() for p in sorted((root / ".github" / "workflows").glob("*.yml"))}


def check(root, release):
    cargo = (root / "Cargo.toml").read_text()
    version = workspace_version(cargo)
    problems = []
    for crate, v in path_dependency_versions(cargo):
        if v != version:
            problems.append(f"{crate} path dependency says {v}, the workspace version is {version}")
    if len(path_dependency_versions(cargo)) != 4:
        problems.append("expected the four path dependencies of the root Cargo.toml")
    locked = lock_versions((root / "Cargo.lock").read_text())
    for crate in WORKSPACE_CRATES:
        if locked.get(crate) != version:
            problems.append(f"Cargo.lock has {crate} {locked.get(crate)}, expected {version} (run `cargo update --workspace`)")
    problems += changelog_problems((root / "CHANGELOG.md").read_text(), version, release)
    problems += pin_problems(workflow_texts(root))
    return version, problems


def status(root):
    cargo = (root / "Cargo.toml").read_text()
    print(f"version (the one place): {workspace_version(cargo)}  [workspace.package] in Cargo.toml")
    print("follows it:")
    for crate, v in path_dependency_versions(cargo):
        print(f"  Cargo.toml  {crate} = {v}")
    for crate, v in sorted(lock_versions((root / "Cargo.lock").read_text()).items()):
        print(f"  Cargo.lock  {crate} = {v}")
    row = COMPARISON_ROW.search((root / "docs" / "comparison.md").read_text())
    print(f"  docs/comparison.md  {row.group(0) if row else 'no Latest release row'}")


def bump(root, version, date):
    if not re.fullmatch(VERSION_RE, version):
        raise SystemExit("version must look like 1.2.3")
    cargo = root / "Cargo.toml"
    cargo.write_text(set_version(cargo.read_text(), version))
    log = root / "CHANGELOG.md"
    log.write_text(roll_changelog(log.read_text(), version, date))
    subprocess.run(["cargo", "update", "--workspace"], cwd=root, check=True)
    print(f"bumped to {version}; now run `python3 -I scripts/release.py check --release`")


def post(root, date):
    version = workspace_version((root / "Cargo.toml").read_text())
    path = root / "docs" / "comparison.md"
    path.write_text(set_comparison_row(path.read_text(), version, date))
    print(f"comparison row set to {version}, {date}")


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "status"
    today = datetime.date.today().isoformat()
    if cmd == "status":
        status(ROOT)
    elif cmd == "check":
        version, problems = check(ROOT, "--release" in argv)
        for p in problems:
            print(f"FAIL {p}")
        print(f"{'ok  ' if not problems else 'FAIL'} release {version}")
        return 1 if problems else 0
    elif cmd == "bump" and len(argv) == 3:
        bump(ROOT, argv[2], today)
    elif cmd == "post":
        post(ROOT, today)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
