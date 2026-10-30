"""Every commit of a pull request from someone outside the exempt list must carry a
`Signed-off-by:` line with the author's own address (the Developer Certificate of
Origin, https://developercertificate.org). Maintainers and bots are listed in
.github/dco-exempt.txt, one fnmatch pattern per line.

  python3 -I scripts/dco_check.py BASE..HEAD

Standard library only.
"""
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEP, END = "\x1f", "\x1e"
SIGNOFF = re.compile(r"^Signed-off-by: .+ <([^<>]+)>\s*$", re.M | re.I)


def parse(log):
    """Commits from `git log --format=%H<US>%P<US>%ae<US>%B<RS>`."""
    commits = []
    for chunk in log.split(END):
        if not chunk.strip():
            continue
        sha, parents, email, body = chunk.strip("\n").split(SEP, 3)
        commits.append({"sha": sha, "merge": len(parents.split()) > 1, "email": email.strip(), "body": body})
    return commits


def exempt(email, patterns):
    return any(fnmatch.fnmatchcase(email.lower(), p.lower()) for p in patterns)


def patterns_from(text):
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def missing(commits, patterns):
    """Commits that need a sign-off and do not have one from their author."""
    out = []
    for c in commits:
        if c["merge"] or exempt(c["email"], patterns):
            continue
        signed = {m.lower() for m in SIGNOFF.findall(c["body"])}
        if c["email"].lower() not in signed:
            out.append(c)
    return out


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    log = subprocess.run(
        ["git", "log", f"--format=%H{SEP}%P{SEP}%ae{SEP}%B{END}", argv[1]],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    patterns = patterns_from((ROOT / ".github" / "dco-exempt.txt").read_text())
    bad = missing(parse(log), patterns)
    for c in bad:
        print(f"FAIL {c['sha'][:7]} by {c['email']}: no `Signed-off-by: Name <{c['email']}>` line (git commit -s)")
    print(f"{'ok  ' if not bad else 'FAIL'} {len(parse(log))} commit(s) checked")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
