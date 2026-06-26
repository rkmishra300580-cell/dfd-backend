"""
detector/result.py
Darpan  -  AnalysisResult + Professional PDF Report Generator
AlgorivX.AI . Darpan Forensic Engine v5

Drop-in replacement: preserves the original AnalysisResult API exactly.
  R = AnalysisResult(job_id)
  R.payload[...] = ...
  R.pdf_text(text, style)
  R.save_graph(fig, filename)
  R.build_pdf()
"""

import os
import io
import math
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from .config import REPORT_FOLDER, TMP_FOLDER

# -- Brand colours -------------------------------------------------------------
C_DARK       = HexColor("#080c12")
C_TEAL       = HexColor("#2dd4bf")
C_WHITE      = HexColor("#ffffff")
C_OFF_WHITE  = HexColor("#f4f6f8")
C_TEXT_DARK  = HexColor("#1a2030")
C_TEXT_MID   = HexColor("#4a5568")
C_TEXT_LIGHT = HexColor("#8a9ab5")
C_BORDER     = HexColor("#dde3ec")

C_V_LOW  = HexColor("#059669")
C_V_MED  = HexColor("#d97706")
C_V_HIGH = HexColor("#dc2626")
C_V_CRIT = HexColor("#7c3aed")

PAGE_W, PAGE_H = A4
MARGIN_L  = 22 * mm
MARGIN_R  = 22 * mm
MARGIN_T  = 18 * mm
MARGIN_B  = 18 * mm
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R
HEADER_H  = 14 * mm
FOOTER_H  = 10 * mm

FONT_B = "Helvetica-Bold"
FONT_R = "Helvetica"


# -- Helpers -------------------------------------------------------------------
def _verdict_color(threat: str):
    t = (threat or "").upper()
    if t in ("MINIMAL", "LOW"):  return C_V_LOW
    if t == "MODERATE":          return C_V_MED
    if t in ("HIGH",):           return C_V_HIGH
    if t == "CRITICAL":          return C_V_CRIT
    return C_TEXT_MID


def _draw_watermark(c):
    c.saveState()
    c.setFillColor(Color(0.88, 0.91, 0.95, alpha=0.15))
    c.setFont(FONT_B, 72)
    c.translate(PAGE_W / 2, PAGE_H / 2)
    c.rotate(45)
    c.drawCentredString(0, 0, "DARPAN")
    c.restoreState()


def _draw_header(c, job_id, page_num, total_pages):
    top = PAGE_H - MARGIN_T
    c.setFillColor(C_DARK)
    c.rect(0, top - HEADER_H, PAGE_W, HEADER_H + MARGIN_T, fill=1, stroke=0)

    c.setFillColor(C_TEAL)
    c.setFont(FONT_R, 6.5)
    c.drawString(MARGIN_L, top - 6.5, "AlgorivX.AI")

    c.setFillColor(C_WHITE)
    c.setFont(FONT_B, 11)
    c.drawString(MARGIN_L, top - 16, "Darpan")

    badge = "FORENSIC ENGINE v5"
    bx = MARGIN_L + 52
    by = top - 14
    bw = c.stringWidth(badge, FONT_R, 6) + 8
    c.setFont(FONT_R, 6)
    c.setStrokeColor(C_TEAL)
    c.setLineWidth(0.5)
    c.rect(bx, by - 2, bw, 9, fill=0, stroke=1)
    c.setFillColor(C_TEAL)
    c.drawString(bx + 4, by + 1, badge)

    c.setFillColor(C_TEXT_LIGHT)
    c.setFont(FONT_R, 7)
    rx = PAGE_W - MARGIN_R
    c.drawRightString(rx, top - 8,  f"Report ID: {job_id}")
    c.drawRightString(rx, top - 17, f"Page {page_num} of {total_pages}")

    c.setStrokeColor(C_TEAL)
    c.setLineWidth(1.2)
    c.line(0, top - HEADER_H, PAGE_W, top - HEADER_H)


def _draw_footer(c, filename, generated_at):
    c.setFillColor(C_OFF_WHITE)
    c.rect(0, 0, PAGE_W, FOOTER_H + MARGIN_B, fill=1, stroke=0)
    c.setStrokeColor(C_BORDER)
    c.setLineWidth(0.5)
    c.line(MARGIN_L, FOOTER_H + MARGIN_B, PAGE_W - MARGIN_R, FOOTER_H + MARGIN_B)
    c.setFillColor(C_TEXT_LIGHT)
    c.setFont(FONT_R, 6.5)
    y = MARGIN_B + 3
    c.drawString(MARGIN_L, y, f"File analysed: {filename}")
    c.drawRightString(PAGE_W - MARGIN_R, y, f"Generated: {generated_at}  .  Confidential  -  AlgorivX.AI")


def _draw_stage_heading(c, y, number, title):
    bar_h = 9 * mm
    c.setFillColor(C_DARK)
    c.roundRect(MARGIN_L, y - bar_h, CONTENT_W, bar_h, 3, fill=1, stroke=0)
    bcx, bcy = MARGIN_L + 10, y - bar_h / 2
    c.setFillColor(C_TEAL)
    c.circle(bcx, bcy, 7, fill=1, stroke=0)
    c.setFillColor(C_DARK)
    c.setFont(FONT_B, 7)
    c.drawCentredString(bcx, bcy - 2.5, str(number))
    c.setFillColor(C_WHITE)
    c.setFont(FONT_B, 9)
    c.drawString(MARGIN_L + 22, bcy - 3, title.upper())
    return y - bar_h - 4 * mm


def _place_graph(c, img_path, x, y, w, h, caption=""):
    border = 1.5 * mm
    c.setFillColor(C_OFF_WHITE)
    c.setStrokeColor(C_BORDER)
    c.setLineWidth(0.5)
    c.roundRect(x - border, y - h - border * 2, w + border * 2, h + border * 2, 3, fill=1, stroke=1)
    try:
        img = ImageReader(img_path)
        c.drawImage(img, x, y - h, width=w, height=h, preserveAspectRatio=True, anchor='c')
    except Exception:
        c.setFillColor(C_BORDER)
        c.rect(x, y - h, w, h, fill=1, stroke=0)
        c.setFillColor(C_TEXT_LIGHT)
        c.setFont(FONT_R, 7)
        c.drawCentredString(x + w / 2, y - h / 2, "[graph unavailable]")
    bottom = y - h - border * 2
    if caption:
        c.setFillColor(C_TEXT_MID)
        c.setFont(FONT_R, 6.5)
        c.drawCentredString(x + w / 2, bottom - 5, caption)
        bottom -= 10
    return bottom - 2 * mm


def _score_bar(c, label, value, x, y, bar_w):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0
    c.setFont(FONT_R, 7)
    c.setFillColor(C_TEXT_MID)
    c.drawString(x, y, label)
    c.drawRightString(x + bar_w, y, f"{value:.1f}%")
    y -= 7
    c.setFillColor(C_BORDER)
    c.rect(x, y, bar_w, 4, fill=1, stroke=0)
    col = C_V_LOW if value < 40 else (C_V_MED if value < 65 else C_V_HIGH)
    c.setFillColor(col)
    c.rect(x, y, (value / 100) * bar_w, 4, fill=1, stroke=0)
    return y - 8


# -- AnalysisResult  -  preserves original API exactly ---------------------------
class AnalysisResult:
    """
    Original API (unchanged):
        R = AnalysisResult(job_id)
        R.payload['key'] = value
        R.pdf_text(text, style='Normal')
        R.save_graph(fig, filename)
        R.build_pdf()
    """

    def __init__(self, job_id: str):
        self.job_id       = job_id
        self.payload      = {"job_id": job_id, "graphs": [], "indicators": [], "stats": [], "stage_scores": {}}
        self._pdf_items   = []   # list of (type, content): ('text', ...) or ('graph', path, caption)
        self._graphs      = []   # list of (path, caption) in insertion order
        self._generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        # Derive paths from config
        self.report_path  = os.path.join(REPORT_FOLDER, f"{job_id}.pdf")
        self.graph_dir    = os.path.join(TMP_FOLDER, job_id)
        os.makedirs(self.graph_dir, exist_ok=True)

    # -- Original methods ------------------------------------------------------

    def pdf_text(self, text: str, style: str = "Normal"):
        """Queue a text paragraph for the PDF (original API)."""
        self._pdf_items.append(("text", text, style))

    def add_stat(self, label: str, value):
        """Append a key/value metric to the stats list (shown in frontend + PDF)."""
        self.payload["stats"].append({"label": label, "value": str(value)})

    def add_indicator(self, text: str):
        """Append a flagged indicator string to the indicators list."""
        self.payload["indicators"].append(text)

    def save_graph(self, filename: str, title: str, description: str = "",
                   important: bool = True):
        """
        Register a graph already saved to disk by the pipeline.
        Pipelines call plt.close(fig) themselves before calling this.

        Signature matches all pipeline calls:
            R.save_graph('name.png', 'Title', 'Description', important=True)
        """
        path = os.path.join(self.graph_dir, filename)
        self._graphs.append((path, title))
        if important:
            self.payload["graphs"].append({
                "title":       title,
                "filename":    filename,
                "description": description,
            })
        return path

    def build_pdf(self):
        """Build the professional PDF and write it to self.report_path."""
        filename    = self.payload.get("filename", "unknown")
        file_type   = self.payload.get("file_type", "")
        score       = float(self.payload.get("final_score", 0))
        threat      = self.payload.get("threat_level", "UNKNOWN")
        verdict     = self.payload.get("verdict", "")
        stage_scores = self.payload.get("stage_scores", {})

        total_pages = self._estimate_pages()
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        c.setTitle(f"Darpan Forensic Report  -  {self.job_id}")
        c.setAuthor("AlgorivX.AI . Darpan Forensic Engine v5")

        page_num = 1

        # -- PAGE 1: Cover + Summary -------------------------------------------
        _draw_watermark(c)
        _draw_header(c, self.job_id, page_num, total_pages)
        _draw_footer(c, filename, self._generated_at)
        self._draw_cover(c, score, threat, verdict, stage_scores, file_type, filename)
        c.showPage()
        page_num += 1

        # -- Graph pages by stage ----------------------------------------------
        groups = self._group_graphs()
        for stage_label, items in groups.items():
            page_num = self._draw_graph_section(c, stage_label, items, page_num, total_pages)

        # -- Final Verdict page ------------------------------------------------
        _draw_watermark(c)
        _draw_header(c, self.job_id, page_num, total_pages)
        _draw_footer(c, filename, self._generated_at)
        self._draw_verdict_page(c, score, threat, stage_scores, verdict)
        c.showPage()

        c.save()
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        with open(self.report_path, "wb") as f:
            f.write(buf.getvalue())

    # -- Internal PDF builders -------------------------------------------------

    def _draw_cover(self, c, score, threat, verdict, stage_scores, file_type, filename):
        v_col = _verdict_color(threat)
        content_top = PAGE_H - MARGIN_T - HEADER_H - 6 * mm

        c.setFillColor(C_TEXT_DARK)
        c.setFont(FONT_B, 16)
        c.drawString(MARGIN_L, content_top, "Forensic Analysis Report")

        c.setFont(FONT_R, 8.5)
        c.setFillColor(C_TEXT_MID)
        c.drawString(MARGIN_L, content_top - 14, f"{file_type}  .  {filename}")

        c.setStrokeColor(C_TEAL)
        c.setLineWidth(1)
        c.line(MARGIN_L, content_top - 20, MARGIN_L + CONTENT_W, content_top - 20)

        y = content_top - 32

        # Meta row
        meta = [("Report ID", self.job_id), ("Generated", self._generated_at),
                ("File", filename), ("Type", file_type)]
        col_w = CONTENT_W / 2
        for i, (label, val) in enumerate(meta):
            mx = MARGIN_L + (i % 2) * col_w
            my = y - (i // 2) * 14
            c.setFont(FONT_R, 6.5)
            c.setFillColor(C_TEXT_LIGHT)
            c.drawString(mx, my, label.upper())
            c.setFont(FONT_R, 7.5)
            c.setFillColor(C_TEXT_DARK)
            c.drawString(mx + 48, my, str(val)[:60])
        y -= 38

        # Score card
        card_h = 34 * mm
        c.setFillColor(C_DARK)
        c.roundRect(MARGIN_L, y - card_h, CONTENT_W, card_h, 5, fill=1, stroke=0)

        c.setFont(FONT_B, 44)
        c.setFillColor(v_col)
        c.drawString(MARGIN_L + 8 * mm, y - 25 * mm, f"{score:.1f}%")

        c.setFont(FONT_R, 7)
        c.setFillColor(C_TEXT_LIGHT)
        c.drawString(MARGIN_L + 8 * mm, y - card_h + 5 * mm, "MANIPULATION PROBABILITY")

        div_x = MARGIN_L + 52 * mm
        c.setStrokeColor(HexColor("#1e2a3a"))
        c.setLineWidth(0.5)
        c.line(div_x, y - card_h + 4 * mm, div_x, y - 4 * mm)

        c.setFont(FONT_B, 18)
        c.setFillColor(v_col)
        c.drawString(div_x + 6 * mm, y - 17 * mm, threat.upper())

        bar_y = y - card_h + 3 * mm
        bar_x = MARGIN_L + 8 * mm
        bar_w = 40 * mm
        c.setFillColor(HexColor("#1e2a3a"))
        c.rect(bar_x, bar_y, bar_w, 3, fill=1, stroke=0)
        c.setFillColor(v_col)
        c.rect(bar_x, bar_y, bar_w * (score / 100), 3, fill=1, stroke=0)

        y -= card_h + 6 * mm

        # Stage breakdown
        c.setFont(FONT_B, 8)
        c.setFillColor(C_TEXT_DARK)
        c.drawString(MARGIN_L, y, "Stage Score Breakdown")
        c.setStrokeColor(C_BORDER)
        c.setLineWidth(0.4)
        c.line(MARGIN_L, y - 3, MARGIN_L + CONTENT_W, y - 3)
        y -= 12

        pairs = []
        if stage_scores:
            for k, v in stage_scores.items():
                if v is not None:
                    pairs.append((k.replace("_", " ").title(), v))
        if not pairs:
            pairs = [("Final Score", score)]

        bw = CONTENT_W - 4 * mm
        for label, val in pairs:
            y = _score_bar(c, label, val, MARGIN_L + 2 * mm, y, bw)

        # Verdict text
        y -= 4 * mm
        if verdict:
            c.setFont(FONT_B, 8)
            c.setFillColor(C_TEXT_DARK)
            c.drawString(MARGIN_L, y, "Verdict")
            c.setStrokeColor(C_BORDER)
            c.setLineWidth(0.4)
            c.line(MARGIN_L, y - 3, MARGIN_L + CONTENT_W, y - 3)
            y -= 12
            c.setFont(FONT_R, 7.5)
            c.setFillColor(C_TEXT_MID)
            words = str(verdict).split()
            line_buf, line_w = [], 0
            for w in words:
                ww = c.stringWidth(w + " ", FONT_R, 7.5)
                if line_w + ww > CONTENT_W and line_buf:
                    c.drawString(MARGIN_L, y, " ".join(line_buf))
                    y -= 10
                    line_buf, line_w = [w], ww
                else:
                    line_buf.append(w); line_w += ww
            if line_buf:
                c.drawString(MARGIN_L, y, " ".join(line_buf))

    def _group_graphs(self):
        groups = {}
        stage_map = [
            ("Stage 1  -  Frequency Domain Analysis",
             ["freq", "fft", "power", "phase", "radial", "band", "channel", "noise_residual", "edge"]),
            ("Stage 2  -  Face Forensic Analysis",
             ["face", "blur", "sharpness"]),
            ("Stage 3  -  Manipulation Analysis",
             ["ela", "prnu", "patch", "copy", "meta", "manipulation"]),
            ("Stage 4  -  Vehicle & Object Damage Analysis",
             ["vehicle", "damage", "shadow", "texture", "boundary", "insurance"]),
            ("Stage 5  -  Deep Learning Detector",
             ["dl", "attention", "patch_var", "fusion", "heatmap"]),
        ]
        unclaimed = list(self._graphs)
        for label, keywords in stage_map:
            matched, remaining = [], []
            for item in unclaimed:
                path, cap = item
                if any(kw in os.path.basename(path).lower() for kw in keywords):
                    matched.append(item)
                else:
                    remaining.append(item)
            if matched:
                groups[label] = matched
            unclaimed = remaining
        if unclaimed:
            groups["Additional Analysis"] = unclaimed
        return groups

    def _draw_graph_section(self, c, stage_label, items, page_num, total_pages):
        COLS  = 2
        G_W   = (CONTENT_W - 4 * mm) / 2
        G_H   = G_W * 0.72
        GAP   = 4 * mm
        HEADING_H = 13 * mm + 4 * mm
        row_h = G_H + GAP + 8
        content_top = PAGE_H - MARGIN_T - HEADER_H - 6 * mm
        avail_first = content_top - HEADING_H - (FOOTER_H + MARGIN_B + 4 * mm)
        avail_rest  = content_top - (FOOTER_H + MARGIN_B + 4 * mm)
        rows_first  = max(1, int(avail_first / row_h))
        rows_rest   = max(1, int(avail_rest  / row_h))

        parts = stage_label.split(" - ", 1)
        num_str   = parts[0].strip().replace("Stage", "").strip() if len(parts) > 1 else "."
        title_str = parts[1].strip() if len(parts) > 1 else stage_label

        i, first = 0, True
        while i < len(items):
            _draw_watermark(c)
            _draw_header(c, self.job_id, page_num, total_pages)
            _draw_footer(c, self.payload.get("filename", ""), self._generated_at)

            y = content_top
            if first:
                y = _draw_stage_heading(c, y, num_str, title_str)
                batch = rows_first * COLS
                first = False
            else:
                c.setFont(FONT_R, 7)
                c.setFillColor(C_TEXT_LIGHT)
                c.drawString(MARGIN_L, y, f"{stage_label}  (continued)")
                y -= 10
                batch = rows_rest * COLS

            for j, (path, cap) in enumerate(items[i:i + batch]):
                col = j % COLS
                row = j // COLS
                gx = MARGIN_L + col * (G_W + GAP)
                gy = y - row * row_h
                _place_graph(c, path, gx, gy, G_W, G_H, cap)

            c.showPage()
            page_num += 1
            i += batch

        return page_num

    def _draw_verdict_page(self, c, score, threat, stage_scores, verdict):
        v_col = _verdict_color(threat)
        content_top = PAGE_H - MARGIN_T - HEADER_H - 10 * mm

        c.setFont(FONT_R, 8)
        c.setFillColor(C_TEXT_LIGHT)
        c.drawCentredString(PAGE_W / 2, content_top, "FINAL VERDICT")
        c.setStrokeColor(C_TEAL)
        c.setLineWidth(1.5)
        c.line(MARGIN_L, content_top - 4, PAGE_W - MARGIN_R, content_top - 4)

        y = content_top - 20
        cx, cy = PAGE_W / 2, y - 38 * mm

        c.setFillColor(C_DARK)
        c.circle(cx, cy, 28 * mm, fill=1, stroke=0)
        c.setFillColor(C_WHITE)
        c.circle(cx, cy, 22 * mm, fill=1, stroke=0)
        c.setFont(FONT_B, 28)
        c.setFillColor(v_col)
        c.drawCentredString(cx, cy - 5, f"{score:.1f}%")
        c.setFont(FONT_R, 6)
        c.setFillColor(C_TEXT_MID)
        c.drawCentredString(cx, cy - 16, "MANIPULATION PROBABILITY")

        y_after = cy - 28 * mm - 8 * mm
        c.setFont(FONT_B, 26)
        c.setFillColor(v_col)
        c.drawCentredString(cx, y_after, threat.upper())
        c.setFont(FONT_R, 8)
        c.setFillColor(C_TEXT_MID)
        c.drawCentredString(cx, y_after - 14, "Threat Level")
        y = y_after - 28

        # Breakdown table
        table_w = 100 * mm
        table_x = (PAGE_W - table_w) / 2
        col_lw  = 65 * mm

        c.setFillColor(C_DARK)
        c.rect(table_x, y - 8, table_w, 9, fill=1, stroke=0)
        c.setFont(FONT_B, 6.5)
        c.setFillColor(C_WHITE)
        c.drawString(table_x + 3, y - 5.5, "Analysis Stage")
        c.drawRightString(table_x + table_w - 3, y - 5.5, "Score")
        y -= 8

        rows = []
        if stage_scores:
            for k, v in stage_scores.items():
                if v is not None:
                    rows.append((k.replace("_", " ").title(), v))
        rows.append(("Final Score", score))

        for k, (label, val) in enumerate(rows):
            bg = C_OFF_WHITE if k % 2 == 0 else C_WHITE
            c.setFillColor(bg)
            c.setStrokeColor(C_BORDER)
            c.setLineWidth(0.3)
            c.rect(table_x, y - 8, table_w, 8, fill=1, stroke=1)
            c.setFont(FONT_R, 6.5)
            c.setFillColor(C_TEXT_DARK)
            c.drawString(table_x + 3, y - 5.5, label)
            try:
                num = float(val)
                disp = f"{num:.1f}%"
                col = C_V_LOW if num < 40 else (C_V_MED if num < 65 else C_V_HIGH)
            except (TypeError, ValueError):
                disp, col = str(val), C_TEXT_DARK
            c.setFillColor(col)
            c.setFont(FONT_B, 6.5)
            c.drawRightString(table_x + table_w - 3, y - 5.5, disp)
            y -= 8

        y -= 12
        disclaimer = (
            "This report was generated automatically by the Darpan Forensic Engine v5 (AlgorivX.AI). "
            "It is intended for use by qualified forensic analysts, security researchers, journalists, "
            "and insurance investigators. Pure colour grading and liquify/warp operations without "
            "structural change may not be detected by any current forensic tool."
        )
        c.setFont(FONT_R, 6)
        c.setFillColor(C_TEXT_LIGHT)
        words = disclaimer.split()
        line_buf, line_w = [], 0
        max_w = CONTENT_W - 20 * mm
        for w in words:
            ww = c.stringWidth(w + " ", FONT_R, 6)
            if line_w + ww > max_w and line_buf:
                c.drawCentredString(PAGE_W / 2, y, " ".join(line_buf))
                y -= 8
                line_buf, line_w = [w], ww
            else:
                line_buf.append(w); line_w += ww
        if line_buf:
            c.drawCentredString(PAGE_W / 2, y, " ".join(line_buf))

        y -= 16
        c.setStrokeColor(C_TEAL)
        c.setLineWidth(0.5)
        sw = 60 * mm
        c.line(PAGE_W / 2 - sw / 2, y, PAGE_W / 2 + sw / 2, y)
        y -= 8
        c.setFont(FONT_R, 6)
        c.setFillColor(C_TEXT_LIGHT)
        c.drawCentredString(PAGE_W / 2, y, f"AlgorivX.AI . Darpan . {self._generated_at}")

    def _estimate_pages(self):
        return 1 + max(1, math.ceil(len(self._graphs) / 4)) + 1
