"""Check the follow-up result tables and figure drawings against the thesis PDF."""

import argparse
from collections import Counter
import csv
from pathlib import Path
import re

import pymupdf


# Row and column counts include group headings in the complete selection table.
TABLE_SHAPES = {
    "3.7": (4, 5),
    "3.8": (5, 5),
    "3.9": (4, 8),
    **{f"A.{n}": (16, 6) for n in range(5, 11)},
    "A.11": (16, 8),
    "A.12": (4, 6),
    "A.13": (11, 6),
    "A.15": (21, 4),
}
FIGURES = {"3.6", "3.7", "A.2"}


def normalize(text):
    return " ".join(text.replace("−", "-").split())


def check_table(number, path, raw, caption):
    with path.open(encoding="utf-8", newline="") as source:
        header, *rows = csv.reader(source)
    height, width = TABLE_SHAPES[number]
    if len(rows) != height or any(len(row) != width for row in [header, *rows]):
        raise ValueError(f"Unexpected table dimensions: {path.name}")

    # The six source-specific tables put their captions above the body. The rest put them below.
    if number in {f"A.{i}" for i in range(5, 11)}:
        section = raw[caption.end() :].split("Table A.")[0]
    else:
        section = raw[: caption.start()]
    body = normalize(" ".join(cell for row in rows for cell in row))
    if body not in normalize(section):
        raise ValueError(f"{path.name} differs from the printed table body")


def check_figure(thesis, page, path):
    ((xref, _, _, bounds),) = [item for item in page.get_xobjects() if item[2] == 0]
    drawing = thesis.xref_stream(xref)
    with pymupdf.open(path) as rebuilt:
        if len(rebuilt) != 1 or not rebuilt[0].rect.is_valid:
            raise ValueError(f"Expected one figure page in {path.name}")
        if max(abs(x - y) for x, y in zip(rebuilt[0].rect, bounds)) > 0.001:
            raise ValueError(f"{path.name} has a different canvas")
        if rebuilt[0].read_contents() != drawing:
            raise ValueError(
                f"{path.name} has different PDF drawing instructions. Check the renderer and fonts."
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    expected = {f"table-{n}.csv" for n in TABLE_SHAPES} | {f"figure-{n}.pdf" for n in FIGURES}
    actual = {f.name for f in args.results.iterdir()}
    if actual != expected:
        raise ValueError(
            f"Expected a default results directory. Missing: {sorted(expected-actual)}. Extra: {sorted(actual-expected)}"
        )
    found = Counter()
    with pymupdf.open(args.pdf) as thesis:
        for page in thesis:
            raw = page.get_text()
            for match in re.finditer(r"^(Table|Figure) ((?:3|A)\.\d+):", raw, re.M):
                kind, number = match.groups()
                if kind == "Table" and number in TABLE_SHAPES:
                    name = f"table-{number}.csv"
                    check_table(number, args.results / name, raw, match)
                elif kind == "Figure" and number in FIGURES:
                    name = f"figure-{number}.pdf"
                    check_figure(thesis, page, args.results / name)
                else:
                    continue
                found[name] += 1
    if found != Counter(expected):
        raise ValueError("Missing or duplicated captions in the reference PDF")
    print(
        f"All {len(TABLE_SHAPES)} table bodies match at printed precision. All {len(FIGURES)} PDF drawing streams match byte for byte."
    )


if __name__ == "__main__":
    main()
