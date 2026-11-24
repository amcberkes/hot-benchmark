"""Compare figure labels and the six source-specific tables with the thesis PDF."""

import argparse
import csv
from collections import Counter
from pathlib import Path
import re

import pymupdf


def figure_page(thesis, index):
    panel = pymupdf.open()
    panel.insert_pdf(thesis, from_page=index, to_page=index)
    page = panel[0]
    ((_, name, _, bounds),) = [item for item in page.get_xobjects() if item[2] == 0]
    page.set_mediabox(pymupdf.Rect(bounds))
    stream = panel.get_new_xref()
    panel.update_object(stream, "<<>>")
    panel.update_stream(stream, f"q /{name} Do Q".encode("ascii"))
    page.set_contents(stream)
    return panel


def labels(page):
    return [
        (text, (x0 + x1) / 2, (y0 + y1) / 2)
        for x0, y0, x1, y1, text, *_ in page.get_text("words")
        if re.search(r"\d", text)
    ]


def check_figure(ref, rebuilt, number):
    expected, actual = labels(ref), labels(rebuilt)
    if Counter(r[0] for r in actual) != Counter(r[0] for r in expected):
        raise ValueError(f"Figure {number} has different numeric labels")
    for value, x, y in expected:
        distance, idx = min(
            (max(abs(px - x), abs(py - y)), idx) for idx, (v, px, py) in enumerate(actual) if v == value
        )
        if distance > 3:
            raise ValueError(f"Figure {number} label {value} differs in position by {distance:.2f} points")
        actual.pop(idx)
    return len(expected)


def check_table(section, path):
    with path.open() as f:
        rows = {r["target"]: r for r in csv.DictReader(f)}
    for match in re.finditer(r"^((?:OF|OS|RF|RS)-[01]{2})$", section, re.M):
        row = rows.pop(match[1])
        values = [
            float(row[k])
            for k in (
                "energy_pct",
                "lower_E",
                "upper_E",
                "discomfort_pct",
                "lower_C",
                "upper_C",
                "discomfort_difference_c2",
            )
        ]
        e, el, eu, c, cl, cu, delta = values
        rendered = (
            f"{e:+.2f} [{el:+.2f}, {eu:+.2f}] {c:+.2f} [{cl:+.2f}, {cu:+.2f}] "
            f"{delta:+.4f} {row['joint_of_10']} {row['comfort_of_10']}"
        )
        printed = " ".join(section[match.end() :].strip().splitlines()[:5]).replace("−", "-")
        if rendered != printed:
            raise ValueError(f"{path.name}, {match[1]}: expected {printed}, generated {rendered}")
    if rows:
        raise ValueError(f"{path.name} has unmatched targets: {list(rows)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    tables, figures = set(), {}
    with pymupdf.open(args.pdf) as thesis:
        for page in thesis:
            raw = page.get_text()
            for m in re.finditer(r"^Figure (3\.[1-7]|A\.[12]):", raw, re.M):
                n = m[1]
                with figure_page(thesis, page.number) as ref, pymupdf.open(
                    args.results / f"figure-{n}.pdf"
                ) as built:
                    figures[n] = check_figure(ref[0], built[0], n)
            for m in re.finditer(r"^Table A\.(5|6|7|8|9|10):", raw, re.M):
                n = int(m[1])
                check_table(raw[m.end() :].split("Table A.")[0], args.results / f"table-A.{n}.csv")
                tables.add(n)
    if len(figures) != 9 or tables != set(range(5, 11)):
        raise ValueError("Expected Figures 3.1–3.7 and A.1–A.2 and Tables A.5–A.10")
    print(
        f"{sum(figures.values())} numeric labels across nine figures and 864 entries in Tables A.5–A.10 match the thesis."
    )


if __name__ == "__main__":
    main()
