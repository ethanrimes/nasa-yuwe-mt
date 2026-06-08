"""Fetch REAL public-domain Bible text (English WEB + Chinese CUV) by verse ref.

We already have, for every Bible verse in the Nasa-Spanish set, the exact
`book / chapter / verse` reference (book as a USFM code, e.g. ``JHN``, ``1CO``).
So instead of machine-translating the Bible we fetch the genuine published
parallel text and inner-join on those refs — far higher quality than any MT.

Source: bible-api.com (public domain). We use its structured data endpoint,
which is keyed by USFM book id and returns a whole chapter at once::

    https://bible-api.com/data/{translation}/{BOOK}/{chapter}

    en -> "web"  (World English Bible)
    zh -> "cuv"  (Chinese Union Version, Traditional / zh-tw)

Because the CUV is Traditional, we optionally convert it to Simplified
(zh-Hans) with OpenCC so the whole reconstructed En-Zh set is one script.

Whole chapters are cached on disk so re-runs / dry-runs are free and offline,
and the entire NT is only ~260 chapters per translation (~520 requests).

CLI:
    python -m nymt_shared.bible chapter JHN 3
    python -m nymt_shared.bible verse JHN 3 16 --pair
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

from . import config

API = "https://bible-api.com/data"
CACHE_DIR = config.REPO_ROOT / "english-chinese-mt-experiment" / "data-en-zh" / "raw" / "bible_cache"

USFM_BOOKS = {
    "GEN", "EXO", "LEV", "NUM", "DEU", "MAT", "MRK", "LUK", "JHN", "ACT",
    "ROM", "1CO", "2CO", "GAL", "EPH", "PHP", "COL", "1TH", "2TH", "1TI",
    "2TI", "TIT", "PHM", "HEB", "JAS", "1PE", "2PE", "1JN", "2JN", "3JN",
    "JUD", "REV",
}

# Only needed so the CLI also accepts a few common English names; the dataset
# itself already stores USFM ids, which the data endpoint consumes directly.
NAME_TO_USFM = {
    "genesis": "GEN", "gen": "GEN",
    "matthew": "MAT", "mat": "MAT", "mt": "MAT",
    "mark": "MRK", "mrk": "MRK", "mk": "MRK",
    "luke": "LUK", "luk": "LUK", "lk": "LUK",
    "john": "JHN", "jhn": "JHN", "jn": "JHN",
    "acts": "ACT", "romans": "ROM", "rom": "ROM",
    "revelation": "REV", "rev": "REV",
}

_T2S = None


def _to_simplified(text: str | None) -> str | None:
    """Convert Traditional Chinese to Simplified (cached OpenCC converter)."""
    global _T2S
    if not text:
        return text
    if _T2S is None:
        try:
            from opencc import OpenCC

            _T2S = OpenCC("t2s")
        except Exception:  # noqa: BLE001 - opencc optional; identity fallback
            _T2S = False
    if _T2S is False:
        return text
    return _T2S.convert(text)


def _canon_book(book: str) -> str:
    """Accept a USFM id (passthrough, upper-cased) or a common English name."""
    b = book.strip()
    if b.upper() in USFM_BOOKS:
        return b.upper()
    return NAME_TO_USFM.get(b.lower(), b.upper())


def _cache_path(book: str, chapter, translation: str) -> Path:
    return CACHE_DIR / translation / book / f"{chapter}.json"


def fetch_chapter(book: str, chapter, translation: str = "web",
                  timeout: float = 20.0, retries: int = 3,
                  pause: float = 0.25) -> dict[int, str]:
    """Return ``{verse_int: text}`` for one chapter. Cached on disk per chapter."""
    usfm = _canon_book(book)
    cp = _cache_path(usfm, chapter, translation)
    if cp.exists():
        data = json.loads(cp.read_text(encoding="utf-8"))
        return {int(k): v for k, v in data.items()}

    url = f"{API}/{translation}/{usfm}/{chapter}"
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 404:
                cp.parent.mkdir(parents=True, exist_ok=True)
                cp.write_text("{}", encoding="utf-8")
                return {}
            r.raise_for_status()
            verses = r.json().get("verses", [])
            out: dict[int, str] = {}
            for v in verses:
                txt = (v.get("text") or "").strip()
                if translation == "cuv":
                    txt = _to_simplified(txt)
                out[int(v["verse"])] = txt
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            time.sleep(pause)
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(pause * (attempt + 1))
    print(f"[bible] WARN failed {translation}/{usfm}/{chapter}: {last_err}",
          file=sys.stderr)
    return {}


def fetch_verse(book: str, chapter, verse, translation: str = "web") -> str | None:
    """Return a single verse's text (str) or None if unavailable. Cached."""
    return fetch_chapter(book, chapter, translation).get(int(verse)) or None


def fetch_pair(book: str, chapter, verse) -> dict | None:
    """Return {'en':..., 'zh':...} for one verse, or None if either side missing."""
    en = fetch_verse(book, chapter, verse, "web")
    zh = fetch_verse(book, chapter, verse, "cuv")
    if en and zh:
        return {"en": en, "zh": zh}
    return None


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="nymt_shared.bible")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("chapter")
    pc.add_argument("book"); pc.add_argument("chapter")
    pc.add_argument("--zh", action="store_true")

    pv = sub.add_parser("verse")
    pv.add_argument("book"); pv.add_argument("chapter"); pv.add_argument("verse")
    pv.add_argument("--zh", action="store_true")
    pv.add_argument("--pair", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "chapter":
        tr = "cuv" if args.zh else "web"
        print(json.dumps(fetch_chapter(args.book, args.chapter, tr),
                         ensure_ascii=False, indent=2))
    elif args.cmd == "verse":
        if args.pair:
            print(json.dumps(fetch_pair(args.book, args.chapter, args.verse),
                             ensure_ascii=False, indent=2))
        else:
            tr = "cuv" if args.zh else "web"
            print(fetch_verse(args.book, args.chapter, args.verse, tr))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
