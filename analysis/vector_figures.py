"""Minimal ReportLab drawing helper shared by the analysis figures.

The paper's analysis figures are vector PDFs drawn at their printed size with Helvetica, so text keeps
its nominal point size. Each PDF stores a JSON payload of the plotted numbers in its Subject field;
``read_payload`` returns it, which is how ``analysis/verify.py`` compares a regenerated figure with the
one in the paper.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

INK, MUTED, RULE, GRID = "#1C2B37", "#405464", "#92A1AF", "#E8EDF1"
METHOD_COLORS = {"LESSER": "#C47A27", "LESS": "#506477", "RDS+": "#8A63B6", "Random": "#858A90"}
BAND = "#D6DBDF"

# Figure 7 panels are drawn at the displayed size (0.49 linewidth, about 194.5 pt).
W7, H7 = 194.5, 148
FT, FY, FX, FL, FLEG = 9, 7.5, 8, 8, 7.8


def _reportlab():
    try:
        from reportlab.lib.colors import HexColor
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover - dependency message
        raise ImportError("figures need reportlab (pip install 'lesser[figures]')") from exc
    return canvas, HexColor, stringWidth


def string_width(text: str, font: str = "Helvetica", size: float = 8) -> float:
    return _reportlab()[2](text, font, size)


class Fig:
    """One-page vector figure with a JSON payload in the PDF Subject."""

    def __init__(self, path, width, height, payload):
        canvas, self._hex, _ = _reportlab()
        path = Path(path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"vector figures need a .pdf path, got {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pdf = canvas.Canvas(str(path), pagesize=(width, height), pageCompression=1)
        self.pdf.setSubject(json.dumps(payload, sort_keys=True))
        self.pdf.setLineCap(1)
        self.pdf.setLineJoin(1)
        self.w, self.h = width, height

    def color(self, value):
        return self._hex(value) if isinstance(value, str) else value

    def text(self, x, y, s, size=8, align="left", bold=False, color=INK):
        p = self.pdf
        p.setFillColor(self.color(color))
        p.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        {"left": p.drawString, "center": p.drawCentredString, "right": p.drawRightString}[align](x, y, str(s))

    def line(self, x0, y0, x1, y1, color=GRID, width=.45, dash=()):
        p = self.pdf
        p.setStrokeColor(self.color(color))
        p.setLineWidth(width)
        p.setDash(*dash) if dash else p.setDash()
        p.line(x0, y0, x1, y1)
        p.setDash()

    def marker(self, x, y, color, shape="circle", r=2.2):
        p = self.pdf
        p.setFillColor(self.color(color))
        p.setStrokeColor(self._hex("#FFFFFF"))
        p.setLineWidth(.5)
        if shape == "square":
            p.rect(x - r, y - r, 2 * r, 2 * r, fill=1, stroke=1)
        else:
            p.circle(x, y, r, fill=1, stroke=1)

    def polyline(self, pts, color, width=1.0, dash=(), alpha=1.0):
        p = self.pdf
        p.saveState()
        p.setStrokeAlpha(alpha)
        p.setStrokeColor(self.color(color))
        p.setLineWidth(width)
        p.setDash(*dash) if dash else p.setDash()
        path = p.beginPath()
        started = False
        for x, y in pts:
            if x is None or y is None or not math.isfinite(y):
                started = False
                continue
            (path.lineTo if started else path.moveTo)(x, y)
            started = True
        p.drawPath(path)
        p.restoreState()

    def vtext(self, x, y, s, size=8, bold=False):
        p = self.pdf
        p.saveState()
        p.translate(x, y)
        p.rotate(90)
        self.text(0, 0, s, size, "center", bold)
        p.restoreState()

    def axes(self, x0, x1, y0, y1):
        self.line(x0, y0, x0, y1, RULE, .55)
        self.line(x0, y0, x1, y0, RULE, .55)

    def save(self):
        self.pdf.showPage()
        self.pdf.save()


def ticks_for(lo, hi, n=3):
    """About n round tick values covering [lo, hi] and the number of decimals to print them with."""
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((mag * s for s in (1, 2, 2.5, 5, 10)), key=lambda s: abs(s - raw))
    first = math.ceil(lo / step - 1e-9) * step
    out, v = [], first
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    digits = max(0, -math.floor(math.log10(step) + 1e-9))
    if any(abs(t * 10 ** digits - round(t * 10 ** digits)) > 1e-6 for t in out):
        digits += 1
    return out, digits


def read_payload(pdf_path) -> dict:
    """The JSON payload stored in a figure's PDF Subject field."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency message
        raise ImportError("reading figure payloads needs pypdf (pip install 'lesser[figures]')") from exc
    meta = PdfReader(str(pdf_path)).metadata
    if meta is None or "/Subject" not in meta:
        raise ValueError(f"{pdf_path} has no payload in its Subject field")
    return json.loads(str(meta["/Subject"]))
