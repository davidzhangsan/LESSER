"""Vector-PDF renderer of the paper's one-row SFT figures (Figures 3, 4 and 8--10), drawn directly with ReportLab.

The same renderer produced the paper's figures, so the same inputs produce the same page content.
The plotted values are embedded as JSON in the PDF's Subject field for provenance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

# Method colors shared by the paper's figures: amber for LESSER, blue-gray for LESS.
PAPER_METHOD_COLORS = {"LESSER": "#C47A27", "LESS": "#506477", "RDS+": "#8A63B6", "Random": "#858A90"}


def paper_main_vector_figure(panels, methods, output, *, ylabel, xlabel, xticks,
                             metadata, endpoint_labels=False, width=396, height=100.8,
                             panel_gap=18, shared_xlabel=False):
    """Render one horizontal row of panels as a vector PDF (dimensions in points).

    ``panels``: dicts with ``title``, ``xlim`` and ``series[method key] = {"x", "y", "sd"}``; a panel may override
    ylim, xticks, xlabel, ylabel, endpoint_labels and reference_y. ``methods``: dicts with ``key``, ``label``,
    ``color`` and optionally ``dash``. ``shared_xlabel`` prints the x-axis label once below the row.
    """
    from reportlab.lib.colors import HexColor
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen import canvas

    if Path(output).suffix.lower() != ".pdf" or not panels or not methods:
        raise ValueError("The vector figure requires PDF output, panels, and methods")
    left, right, bottom, top = 34, 4, 26.5, 31
    panel_width = (width-left-right-(len(panels)-1)*panel_gap)/len(panels)
    panel_height = height-top-bottom
    if panel_width <= 0 or panel_height <= 0:
        raise ValueError("The requested canvas leaves no space for the data panels")
    ink, muted, rule, grid = map(HexColor, ("#1C2B37", "#405464", "#92A1AF", "#E8EDF1"))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output), pagesize=(width, height), pageCompression=1)
    pdf.setSubject(json.dumps(metadata))
    pdf.setLineCap(1)
    pdf.setLineJoin(1)

    def text(x, y, value, size=8, align="left", color=ink, bold=False):
        pdf.setFillColor(color)
        pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        {"left": pdf.drawString, "center": pdf.drawCentredString,
         "right": pdf.drawRightString}[align](x, y, str(value))

    def style(index):
        return (4, 2.4) if index == 1 else (1, 1.7) if index == 3 else ()

    legend_widths = [17+stringWidth(m["label"], "Helvetica", 8.5) for m in methods]
    if sum(legend_widths)+14*(len(methods)-1) > width-8:
        raise ValueError("The method legend does not fit the requested canvas width")
    x = (width-sum(legend_widths)-14*(len(methods)-1))/2
    for index, (method, item_width) in enumerate(zip(methods, legend_widths)):
        color = HexColor(method["color"])
        pdf.setStrokeColor(color); pdf.setLineWidth(1.2); pdf.setDash(method.get("dash", style(index)))
        pdf.line(x, height-10, x+13, height-10)
        text(x+17, height-12.6, method["label"], 8.5)
        x += item_width+14
    pdf.setDash()

    for panel_index, panel in enumerate(panels):
        x0 = left+panel_index*(panel_width+panel_gap)
        xlow, xhigh = panel["xlim"]
        lows, highs = [], []
        for method in methods:
            series = panel["series"][method["key"]]
            xs, ys, errors = series["x"], series["y"], series["sd"]
            if len(xs) != len(ys) or len(errors) != len(ys) or len(xs) < 2:
                raise ValueError("Each plotted series needs matched x, y, and SD arrays")
            if not all(math.isfinite(v) for v in [*xs, *ys, *errors]) or any(v < 0 for v in errors):
                raise ValueError("Plot values must be finite and SDs nonnegative")
            lows.extend(y-e for y, e in zip(ys, errors))
            highs.extend(y+e for y, e in zip(ys, errors))
        data_span = max(highs)-min(lows)
        if data_span <= 0 or xhigh <= xlow:
            raise ValueError("The main plot requires nonzero data ranges")
        ylow, yhigh = panel.get("ylim", (min(lows)-.10*data_span, max(highs)+.10*data_span))
        if not math.isfinite(ylow) or not math.isfinite(yhigh) or yhigh <= ylow:
            raise ValueError("Each panel requires finite, increasing y limits")
        raw_step = (yhigh-ylow)/3
        magnitude = 10**math.floor(math.log10(raw_step))
        step = min((magnitude*s for s in (1, 2, 5, 10)), key=lambda s: abs(s-raw_step))
        digits = max(0, -math.floor(math.log10(step)))
        if not math.isclose(step*10**digits, round(step*10**digits)):
            digits += 1
        ticks = [i*step for i in range(math.ceil(ylow/step), math.floor(yhigh/step)+1)]
        xp = lambda value: x0+(value-xlow)/(xhigh-xlow)*panel_width  # noqa: E731
        yp = lambda value: bottom+(value-ylow)/(yhigh-ylow)*panel_height  # noqa: E731
        pdf.setDash()
        for value in ticks:
            y = yp(value)
            pdf.setStrokeColor(grid); pdf.setLineWidth(.45)
            pdf.line(x0, y, x0+panel_width, y)
            text(x0-3, y-2.6, f"{value:.{digits}f}", 7.5, "right", muted)
        for value in panel.get("reference_y", []):
            if not math.isfinite(value):
                raise ValueError("Reference lines require finite y values")
            if ylow <= value <= yhigh:
                pdf.setStrokeColor(rule); pdf.setLineWidth(.7); pdf.setDash(2, 2)
                pdf.line(x0, yp(value), x0+panel_width, yp(value))
        pdf.setDash()
        pdf.setStrokeColor(rule); pdf.setLineWidth(.55)
        pdf.line(x0, bottom, x0, bottom+panel_height)
        pdf.line(x0, bottom, x0+panel_width, bottom)
        for tick_index, (value, label) in enumerate(panel.get("xticks", xticks)):
            x = xp(value)
            pdf.line(x, bottom, x, bottom-2)
            align = (("left" if tick_index == 0 else "right")
                     if panel.get("endpoint_labels", endpoint_labels) else "center")
            text(x, bottom-11.5, label, 8, align, muted)
        if not shared_xlabel:
            text(x0+panel_width/2, bottom-21, panel.get("xlabel", xlabel), 8, "center")
        text(x0+panel_width/2, bottom+panel_height+6, panel["title"], 9, "center")
        panel_ylabel = panel.get("ylabel", ylabel if panel_index == 0 else "")
        if panel_ylabel:
            pdf.saveState(); pdf.translate(x0-25, bottom+panel_height/2); pdf.rotate(90)
            text(0, 0, panel_ylabel, 8, "center"); pdf.restoreState()
        pdf.saveState()
        clip = pdf.beginPath(); clip.rect(x0, bottom, panel_width, panel_height)
        pdf.clipPath(clip, stroke=0, fill=0)
        for index, method in enumerate(methods):
            series = panel["series"][method["key"]]
            color = HexColor(method["color"])
            pdf.setStrokeColor(color); pdf.setLineWidth(.65); pdf.setDash()
            for x, y, error in zip(series["x"], series["y"], series["sd"]):
                if error:
                    pdf.line(xp(x), yp(y-error), xp(x), yp(y+error))
                    pdf.line(xp(x)-1.4, yp(y-error), xp(x)+1.4, yp(y-error))
                    pdf.line(xp(x)-1.4, yp(y+error), xp(x)+1.4, yp(y+error))
            pdf.setLineWidth(1.2 if index == 0 else 1.05); pdf.setDash(method.get("dash", style(index)))
            path = pdf.beginPath(); path.moveTo(xp(series["x"][0]), yp(series["y"][0]))
            for x, y in zip(series["x"][1:], series["y"][1:]):
                path.lineTo(xp(x), yp(y))
            pdf.drawPath(path)
        pdf.restoreState()
    if shared_xlabel:
        text(left+(width-left-right)/2, bottom-21, xlabel, 8, "center")
    pdf.showPage(); pdf.save()


def pdf_page_content(path) -> bytes:
    """Decompressed drawing operators of page 1 (independent of the PDF's dates and metadata)."""
    from pypdf import PdfReader
    return PdfReader(str(path)).pages[0].get_contents().get_data()


def pdf_payload(path) -> dict:
    """The JSON payload stored in a figure's Subject field."""
    from pypdf import PdfReader
    subject = PdfReader(str(path)).metadata.get("/Subject")
    if not subject:
        raise ValueError(f"{path}: no embedded payload")
    return json.loads(subject)
