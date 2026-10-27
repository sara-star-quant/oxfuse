#!/usr/bin/env python3
"""Fetch every https URL in docs/comparison.md and report the ones that do not answer 200.

Run by hand before a release; it needs the network, so CI does not run it.
  python3 -I scripts/check_links.py
Stdlib only.
"""

import re
import sys
import urllib.request
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "docs" / "comparison.md"


def status(url):
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": "oxfuse-link-check"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except Exception as error:  # report, do not stop
        return getattr(error, "code", str(error))


def main():
    urls = sorted(set(re.findall(r"<(https://[^>]+)>", PAGE.read_text())))
    bad = 0
    for url in urls:
        code = status(url.split("#")[0])
        print(code, url)
        bad += code != 200
    print(f"{len(urls)} links, {bad} not 200")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
