"""Rewrite the feature table in BRIDGE-Models.drawio's output boxes from the live feature table.

The table in each page's "Phonology (features)" box shows the decoder's training targets for
"cat": which phonetic features are on at each position. `tests/test_diagram.py` checks it
against `bridge/core/phonreps.csv` and fails when they disagree, so after editing the CSV run

    uv run python tests/fixtures/sync_diagram_table.py

and the test passes again. Only the table's own cells (ids ending `-out-<n>`) are rewritten;
everything else in the file stays byte-identical. Running it on an up-to-date file changes
nothing. The values come from `cat_targets` in the test, so the two cannot disagree.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.test_diagram import DIAGRAM, cat_targets  # noqa: E402

FONT = "fontFamily=Times New Roman;"
ON, MUTED, OFF_DIGIT = "#1B5E20", "#5F6F5F", "#9AA79A"


def short(name: str) -> str:
    return name.replace("[EOS]", "EOS")


def table_cells(prefix: str, group: str, fade: str) -> list[str]:
    """The table's cells, one line each, in drawing order. Layout in the box's local units."""
    names, phonemes, targets = cat_targets()
    rows = [n for i, n in enumerate(names) if any(t[i] for t in targets)]
    cells: list[str] = []

    def add(x: float, y: float, w: float, h: float, style: str, value: str = "") -> None:
        cells.append(
            f'<mxCell id="{prefix}-out-{len(cells) + 1}" value="{escape(value, {chr(34): "&quot;"})}" '
            f'style="{style}{fade}" vertex="1" parent="{group}"><mxGeometry x="{x:g}" y="{y:g}" '
            f'width="{w:g}" height="{h:g}" as="geometry" /></mxCell>'
        )

    def text(x, y, w, h, value, size=8, bold=True, align="center", color="#000000", extra=""):
        add(
            x,
            y,
            w,
            h,
            f"text;html=1;whiteSpace=wrap;align={align};verticalAlign=middle;spacing=0;"
            f"{FONT}fontSize={size};fontStyle={1 if bold else 0};fontColor={color};{extra}",
            value,
        )

    x_names, x0, cw, y0, rh = 0, 39, 11, 60, 14
    for j, phoneme in enumerate(phonemes):
        text(x0 + j * cw - 2, 47, cw + 4, 10, short(phoneme), size=7)
        add(
            x0 + j * cw + 1,
            y0 - 2,
            cw - 2,
            rh * len(rows) + 4,
            "rounded=1;arcSize=20;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#1B5E20;strokeWidth=1;",
        )
    for i, name in enumerate(rows):
        y = y0 + i * rh
        text(
            x_names,
            y,
            38,
            rh,
            short(name),
            size=7,
            bold=False,
            align="right",
            extra="spacingRight=2;",
        )
        for j in range(len(phonemes)):
            bit = targets[j][names.index(name)]
            text(
                x0 + j * cw,
                y,
                cw,
                rh,
                str(bit),
                size=8,
                bold=bool(bit),
                color=ON if bit else OFF_DIGIT,
            )
    y = y0 + len(rows) * rh + 4
    text(
        2,
        y,
        86,
        22,
        f"+ {len(names) - len(rows)} more features,<br>all 0 here",
        size=7,
        bold=False,
        color=MUTED,
    )
    text(
        2, y + 24, 86, 14, "each column &isin; {0,1}<sup>35</sup>", size=7, bold=False, color=MUTED
    )
    return cells


def _attrs(tag: str) -> dict[str, str]:
    return dict(re.findall(r'(\w+)="([^"]*)"', tag))


def _meaning(cell_xml: str) -> tuple:
    """A cell as draw.io understands it, independent of how it was serialized: attribute order,
    line breaks, and the omitted defaults draw.io drops on save (x or y of 0)."""
    el = ET.fromstring(cell_xml)
    geo = el.find("mxGeometry")
    g = {k: float(v) for k, v in (geo.attrib if geo is not None else {}).items() if k != "as"}
    return (
        el.get("id"),
        el.get("value", ""),
        frozenset(p for p in (el.get("style") or "").split(";") if p),
        el.get("parent"),
        tuple(g.get(k, 0.0) for k in ("x", "y", "width", "height")),
    )


def sync(path: Path) -> int:
    """Rewrite every page's table in place. Returns the number of pages whose table changed.

    Works on files saved by draw.io as well as by this script: cells are found across line
    breaks and in any attribute order, and a page is left untouched when its table already
    means the same thing as the one that would be written."""
    text = path.read_text(encoding="utf-8")
    changed = 0
    containers = [
        a
        for tag in re.findall(r"<mxCell\b[^>]*>", text)
        if (a := _attrs(tag)).get("value") == "Phonology&lt;br&gt;(features)"
    ]
    for a in containers:
        cid, style, group = a["id"], a.get("style", ""), a["parent"]
        prefix = cid.rsplit("-", 1)[0]
        fade = "opacity=20;textOpacity=20;" if "opacity=20" in style else ""
        old = re.compile(
            rf'\n[ \t]*<mxCell id="{re.escape(prefix)}-out-\d+"[^>]*?(?:/>|>.*?</mxCell>)', re.S
        )
        existing = [m.group(0).strip() for m in old.finditer(text)]
        wanted = table_cells(prefix, group, fade)
        if [_meaning(c) for c in existing] == [_meaning(c) for c in wanted]:
            continue
        stripped = old.sub("", text)
        start = stripped.index(f'<mxCell id="{cid}"')
        head_end = stripped.index(">", start)
        end = (
            head_end + 1
            if stripped[head_end - 1] == "/"
            else (stripped.index("</mxCell>", start) + len("</mxCell>"))
        )
        indent = re.search(r"\n([ \t]*)$", stripped[:start]).group(1)
        block = "".join(f"\n{indent}{cell}" for cell in wanted)
        text = stripped[:end] + block + stripped[end:]
        changed += 1
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DIAGRAM
    n = sync(target)
    print(
        f"{target.name}: feature table rewritten on {n} page(s)"
        if n
        else f"{target.name}: already up to date"
    )
