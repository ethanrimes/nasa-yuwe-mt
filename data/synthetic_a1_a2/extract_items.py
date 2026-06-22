"""Extract A1+A2 coverage items from PCIC inventory pages.

Produces items_a1_a2.csv with one row per coverage item:
  cefr_level     A1 | A2   (from the table's A1/A2 column)
  kind           grammar | vocab
  section        gramatica | nociones_generales | nociones_especificas
  concept_group  coarse subsection heading (table caption) -> used for UI filter
  concept        exact PCIC leaf (grammar structure, or single lexical unit)

Grammar pages: each leaf <li> in an A1/A2 column cell = one grammar concept.
Nociones pages: each leaf <li> is a comma/slash separated list of lexical units;
                each lexical unit becomes its own vocab concept.
"""
import csv
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

PCIC = Path(r"C:\Users\ethankallett\Downloads\l2_sources\pcic")
OUT = Path(__file__).with_name("items_a1_a2.csv")
SECTIONS = ["gramatica", "nociones_generales", "nociones_especificas"]


def ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def content(path: Path):
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
    return soup.find(id="contenido") or soup


def leaf_lis(node):
    out = []
    for li in node.find_all("li"):
        if li.find("li"):
            continue
        if li.find_parent(class_="flechas") or li.find_parent(class_="flechas2"):
            continue
        t = ws(li.get_text(" ", strip=True))
        if len(t) > 2:
            out.append(t)
    return out


def lexical_units(txt: str):
    """Split a nociones <li> line into individual lexical units."""
    t = re.sub(r"\[[^\]]*\]", " ", txt)          # drop [region] tags
    t = re.sub(r"\([^)]*\)", " ", t)             # drop (parentheticals)
    if "~" in t:                                  # collocation frame: keep fillers
        t = t.split("~", 1)[1]
    units = []
    for p in re.split(r"[,;/]| o | y ", t):
        p = p.strip(" .:·-\u2013\u2014")
        if len(p) >= 2 and re.search(r"[a-záéíóúñü]", p, re.I):
            units.append(p)
    return units


def col_label(table):
    """Map each leading <td> column index to its A1/A2 header label."""
    ths = [ws(th.get_text(" ", strip=True)) for th in table.find_all("th")]
    idx2lab = {}
    for i, h in enumerate(ths[:2]):
        if h in ("A1", "A2"):
            idx2lab[i] = h
    return idx2lab


def group_for(table):
    cap = table.find("caption")
    if cap:
        return ws(cap.get_text(" ", strip=True))
    for prev in table.find_all_previous(["h4", "h3", "h2"]):
        t = ws(prev.get_text(" ", strip=True))
        if t:
            return t
    return ""


def main():
    rows = []
    for section in SECTIONS:
        cont = content(PCIC / f"{section}_a1-a2.htm")
        is_vocab = section.startswith("nociones")
        for table in cont.find_all("table"):
            idx2lab = col_label(table)
            if not idx2lab:
                continue
            group = group_for(table)
            body = table.find("tbody") or table
            for tr in body.find_all("tr"):
                tds = tr.find_all("td", recursive=False) or tr.find_all("td")
                for i, td in enumerate(tds[:2]):
                    lab = idx2lab.get(i)
                    if not lab:
                        continue
                    for leaf in leaf_lis(td):
                        if is_vocab:
                            for unit in lexical_units(leaf):
                                rows.append((lab, "vocab", section, group, unit))
                        else:
                            rows.append((lab, "grammar", section, group, leaf))

    # de-dup identical (level, kind, section, concept) keeping first group
    seen = set()
    uniq = []
    for r in rows:
        key = (r[0], r[1], r[2], r[4].lower())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "cefr_level", "kind", "section", "concept_group", "concept"])
        for n, r in enumerate(uniq, 1):
            w.writerow([f"pcic-{n:05d}", *r])

    # summary
    from collections import Counter
    by = Counter((r[0], r[1]) for r in uniq)
    print(f"wrote {len(uniq)} items -> {OUT}")
    for (lvl, kind), c in sorted(by.items()):
        print(f"  {lvl} {kind:8} {c}")
    print(f"  raw rows before dedup: {len(rows)}")


if __name__ == "__main__":
    sys.exit(main())
