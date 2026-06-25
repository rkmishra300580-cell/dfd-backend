"""
detector/result.py
Darpan — Professional Forensic Report Generator
AlgorivX.AI · Darpan Forensic Engine v5
"""

import os
import io
import math
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfgen import canvas
from reportlab.platypus import Image as RLImage
from reportlab.lib.utils import ImageReader

# ── Brand colours ────────────────────────────────────────────────────────────
C_DARK        = HexColor("#080c12")       # near-black brand bg
C_TEAL        = HexColor("#2dd4bf")       # primary accent
C_WHITE       = HexColor("#ffffff")
C_OFF_WHITE   = HexColor("#f4f6f8")       # section bg tint
C_TEXT_DARK   = HexColor("#1a2030")       # body text on light
C_TEXT_MID    = HexColor("#4a5568")       # secondary text
C_TEXT_LIGHT  = HexColor("#8a9ab5")       # captions / metadata
C_BORDER      = HexColor("#dde3ec")       # hairline borders
C_VERDICT_LOW    = HexColor("#059669")
C_VERDICT_MED    = HexColor("#d97706")
C_VERDICT_HIGH   = HexColor("#dc2626")
C_VERDICT_CRIT   = HexColor("#7c3aed")

PAGE_W, PAGE_H = A4                        # 595.27 x 841.89 pt
MARGIN_L  = 22 * mm
MARGIN_R  = 22 * mm
MARGIN_T  = 18 * mm
MARGIN_B  = 18 * mm
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

HEADER_H  = 14 * mm
FOOTER_H  = 10 * mm

FONT_BRAND   = "Helvetica-Bold"
FONT_HEADING = "Helvetica-Bold"
FONT_BODY    = "Helvetica"
FONT_MONO    = "Courier"


# ── Watermark ─────────────────────────────────────────────────────────────────
def _draw_watermark(c: canvas.Canvas):
    """Diagonal DARPAN watermark, subtle, every page."""
    c.saveState()
    c.setFillColor(Color(0.88, 0.91, 0.95, alpha=0.18))
    c.setFont(FONT_BRAND, 72)
    c.translate(PAGE_W / 2, PAGE_H / 2)
    c.rotate(45)
    c.drawCentredString(0, 0, "DARPAN")
    c.restoreState()


# ── Header ────────────────────────────────────────────────────────────────────
def _draw_header(c: canvas.Canvas, job_id: str, page_num: int, total_pages: int):
    """Consistent header: dark band with branding left, job ID right."""
    top = PAGE_H - MARGIN_T
    # Dark band
    c.setFillColor(C_DARK)
    c.rect(0, top - HEADER_H, PAGE_W, HEADER_H + MARGIN_T, fill=1, stroke=0)

    # AlgorivX.AI (small, teal)
    c.setFillColor(C_TEAL)
    c.setFont(FONT_BODY, 6.5)
    c.drawString(MARGIN_L, top - 6.5, "AlgorivX.AI")

    # Darpan (larger, white)
    c.setFillColor(C_WHITE)
    c.setFont(FONT_BRAND, 11)
    c.drawString(MARGIN_L, top - 16, "Darpan")

    # Badge
    badge_text = "FORENSIC ENGINE v5"
    c.setFont(FONT_BODY, 6)
    c.setFillColor(HexColor("#2dd4bf"))
    badge_x = MARGIN_L + 52
    badge_y = top - 14
    bw = c.stringWidth(badge_text, FONT_BODY, 6) + 8
    c.setStrokeColor(C_TEAL)
    c.setLineWidth(0.5)
    c.rect(badge_x, badge_y - 2, bw, 9, fill=0, stroke=1)
    c.setFillColor(C_TEAL)
    c.drawString(badge_x + 4, badge_y + 1, badge_text)

    # Job ID + page right-aligned
    c.setFillColor(C_TEXT_LIGHT)
    c.setFont(FONT_BODY, 7)
    right_x = PAGE_W - MARGIN_R
    c.drawRightString(right_x, top - 8, f"Report ID: {job_id}")
    c.drawRightString(right_x, top - 17, f"Page {page_num} of {total_pages}")

    # Teal accent line under header
    c.setStrokeColor(C_TEAL)
    c.setLineWidth(1.2)
    y_line = top - HEADER_H
    c.line(0, y_line, PAGE_W, y_line)


# ── Footer ────────────────────────────────────────────────────────────────────
def _draw_footer(c: canvas.Canvas, filename: str, generated_at: str):
    """Consistent footer: light band with filename and timestamp."""
    c.setFillColor(C_OFF_WHITE)
    c.rect(0, 0, PAGE_W, FOOTER_H + MARGIN_B, fill=1, stroke=0)

    c.setStrokeColor(C_BORDER)
    c.setLineWidth(0.5)
    c.line(MARGIN_L, FOOTER_H + MARGIN_B, PAGE_W - MARGIN_R, FOOTER_H + MARGIN_B)

    c.setFillColor(C_TEXT_LIGHT)
    c.setFont(FONT_BODY, 6.5)
    y = MARGIN_B + 3
    c.drawString(MARGIN_L, y, f"File analysed: {filename}")
    c.drawRightString(PAGE_W - MARGIN_R, y, f"Generated: {generated_at}  ·  Confidential — AlgorivX.AI")


# ── Section heading bar ───────────────────────────────────────────────────────
def _draw_stage_heading(c: canvas.Canvas, y: float, number: str, title: str) -> float:
    """
    Dark pill with stage number + title. Returns y after the heading.
    """
    bar_h = 9 * mm
    # Dark background bar
    c.setFillColor(C_DARK)
    c.roundRect(MARGIN_L, y - bar_h, CONTENT_W, bar_h, 3, fill=1, stroke=0)

    # Teal stage number bubble
    bubble_r = 7
    bubble_cx = MARGIN_L + 10
    bubble_cy = y - bar_h / 2
    c.setFillColor(C_TEAL)
    c.circle(bubble_cx, bubble_cy, bubble_r, fill=1, stroke=0)
    c.setFillColor(C_DARK)
    c.setFont(FONT_BRAND, 7)
    c.drawCentredString(bubble_cx, bubble_cy - 2.5, number)

    # Title text
    c.setFillColor(C_WHITE)
    c.setFont(FONT_HEADING, 9)
    c.drawString(MARGIN_L + 22, bubble_cy - 3, title.upper())

    return y - bar_h - 4 * mm


# ── Graph image helper ────────────────────────────────────────────────────────
def _place_graph(c: canvas.Canvas, img_path: str, x: float, y: float,
                 w: float, h: float, caption: str = ""):
    """
    Draw one graph with a thin border and optional caption.
    y is the TOP of the graph block. Returns y below the caption.
    """
    border = 1.5 * mm
    # Card background
    c.setFillColor(C_OFF_WHITE)
    c.setStrokeColor(C_BORDER)
    c.setLineWidth(0.5)
    c.roundRect(x - border, y - h - border * 2, w + border * 2, h + border * 2, 3, fill=1, stroke=1)

    # Image
    try:
        img = ImageReader(img_path)
        c.drawImage(img, x, y - h, width=w, height=h, preserveAspectRatio=True, anchor='c')
    except Exception:
        c.setFillColor(C_BORDER)
        c.rect(x, y - h, w, h, fill=1, stroke=0)
        c.setFillColor(C_TEXT_LIGHT)
        c.setFont(FONT_BODY, 7)
        c.drawCentredString(x + w / 2, y - h / 2, "[graph unavailable]")

    bottom = y - h - border * 2
    if caption:
        c.setFillColor(C_TEXT_MID)
        c.setFont(FONT_BODY, 6.5)
        c.drawCentredString(x + w / 2, bottom - 5, caption)
        bottom -= 10
    return bottom - 2 * mm


# ── Verdict colour ────────────────────────────────────────────────────────────
def _verdict_color(threat: str) -> Color:
    t = threat.upper()
    if t == "MINIMAL":   return C_VERDICT_LOW
    if t == "LOW":       return C_VERDICT_LOW
    if t == "MEDIUM":    return C_VERDICT_MED
    if t == "HIGH":      return C_VERDICT_HIGH
    if t == "CRITICAL":  return C_VERDICT_CRIT
    return C_TEXT_MID


# ── Horizontal metric bar ─────────────────────────────────────────────────────
def _draw_score_bar(c: canvas.Canvas, label: str, value: float,
                    x: float, y: float, bar_w: float):
    """Single labelled score bar. Returns y below."""
    bar_h = 4
    # label
    c.setFont(FONT_BODY, 7)
    c.setFillColor(C_TEXT_MID)
    c.drawString(x, y, label)
    # value text
    c.drawRightString(x + bar_w, y, f"{value:.1f}%")
    y -= 7
    # track
    c.setFillColor(C_BORDER)
    c.rect(x, y, bar_w, bar_h, fill=1, stroke=0)
    # fill
    fill_w = (value / 100) * bar_w
    col = C_VERDICT_LOW if value < 40 else (C_VERDICT_MED if value < 65 else C_VERDICT_HIGH)
    c.setFillColor(col)
    c.rect(x, y, fill_w, bar_h, fill=1, stroke=0)
    return y - 8


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

class ReportBuilder:
    """
    Drop-in replacement for the old R helper.
    Usage:
        R = ReportBuilder(job_id, filename, file_type, graphs_dir)
        R.save_graph(fig, "name.png")
        pdf_bytes = R.build_pdf(result_dict)
    """

    def __init__(self, job_id: str, filename: str, file_type: str, graphs_dir: str):
        self.job_id     = job_id
        self.filename   = filename
        self.file_type  = file_type.upper()
        self.graphs_dir = graphs_dir
        self.graphs     = []                 # list of (path, caption)
        self.generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        self._buf = io.BytesIO()

    # ── Graph saver ───────────────────────────────────────────────────────────
    def save_graph(self, fig, name: str, caption: str = "") -> str:
        """Save a matplotlib figure to disk. Returns the file path."""
        import matplotlib
        matplotlib.use("Agg")
        path = os.path.join(self.graphs_dir, name)
        fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
        import matplotlib.pyplot as plt
        plt.close(fig)
        self.graphs.append((path, caption or os.path.splitext(name)[0].replace("_", " ").title()))
        return path

    # ── Main builder ──────────────────────────────────────────────────────────
    def build_pdf(self, result: dict) -> bytes:
        """
        Build and return the complete professional PDF as bytes.
        result keys expected (all optional — graceful degradation):
            score, threat, stage_scores, fusion_mode,
            manipulation_score, forensic_score, dl_score,
            vehicle_score, metadata_score, notes
        """
        self._buf = io.BytesIO()
        total_pages = self._count_pages()
        c = canvas.Canvas(self._buf, pagesize=A4)
        c.setTitle(f"Darpan Forensic Report — {self.job_id}")
        c.setAuthor("AlgorivX.AI · Darpan Forensic Engine v5")
        c.setSubject(f"{self.file_type} Analysis — {self.filename}")

        page_num = 1

        # ── PAGE 1: Cover + Summary ───────────────────────────────────────────
        _draw_watermark(c)
        _draw_header(c, self.job_id, page_num, total_pages)
        _draw_footer(c, self.filename, self.generated_at)
        y = self._draw_cover(c, result)
        c.showPage()
        page_num += 1

        # ── GRAPH PAGES ───────────────────────────────────────────────────────
        # Group graphs into stage sections based on naming conventions
        stage_groups = self._group_graphs_by_stage()
        for stage_label, items in stage_groups.items():
            page_num = self._draw_graph_section(
                c, stage_label, items, page_num, total_pages
            )

        # ── FINAL VERDICT PAGE ────────────────────────────────────────────────
        _draw_watermark(c)
        _draw_header(c, self.job_id, page_num, total_pages)
        _draw_footer(c, self.filename, self.generated_at)
        self._draw_verdict_page(c, result)
        c.showPage()

        c.save()
        return self._buf.getvalue()

    # ── Cover / Summary ───────────────────────────────────────────────────────
    def _draw_cover(self, c: canvas.Canvas, result: dict) -> float:
        content_top = PAGE_H - MARGIN_T - HEADER_H - 6 * mm

        # ── Report title block ────────────────────────────────────────────────
        c.setFillColor(C_TEXT_DARK)
        c.setFont(FONT_HEADING, 16)
        c.drawString(MARGIN_L, content_top, "Forensic Analysis Report")

        c.setFont(FONT_BODY, 8.5)
        c.setFillColor(C_TEXT_MID)
        c.drawString(MARGIN_L, content_top - 14,
                     f"{self.file_type} · {self.filename}")

        # Teal rule under title
        c.setStrokeColor(C_TEAL)
        c.setLineWidth(1)
        c.line(MARGIN_L, content_top - 20, MARGIN_L + CONTENT_W, content_top - 20)

        y = content_top - 32

        # ── Meta info row ─────────────────────────────────────────────────────
        meta_items = [
            ("Report ID",   self.job_id),
            ("Generated",   self.generated_at),
            ("File",        self.filename),
            ("Type",        self.file_type),
        ]
        col_w = CONTENT_W / 2
        for i, (label, val) in enumerate(meta_items):
            col = i % 2
            row = i // 2
            mx = MARGIN_L + col * col_w
            my = y - row * 14
            c.setFont(FONT_BODY, 6.5)
            c.setFillColor(C_TEXT_LIGHT)
            c.drawString(mx, my, label.upper())
            c.setFont(FONT_BODY, 7.5)
            c.setFillColor(C_TEXT_DARK)
            c.drawString(mx + 48, my, val)

        y -= 38

        # ── Score summary card ────────────────────────────────────────────────
        score   = float(result.get("score", 0))
        threat  = result.get("threat", "UNKNOWN")
        v_col   = _verdict_color(threat)

        card_h  = 34 * mm
        c.setFillColor(C_DARK)
        c.roundRect(MARGIN_L, y - card_h, CONTENT_W, card_h, 5, fill=1, stroke=0)

        # Big score number
        c.setFont(FONT_BRAND, 44)
        c.setFillColor(v_col)
        c.drawString(MARGIN_L + 8 * mm, y - 25 * mm, f"{score:.1f}%")

        # Label
        c.setFont(FONT_BODY, 7)
        c.setFillColor(C_TEXT_LIGHT)
        c.drawString(MARGIN_L + 8 * mm, y - card_h + 5 * mm, "MANIPULATION PROBABILITY")

        # Vertical divider
        div_x = MARGIN_L + 52 * mm
        c.setStrokeColor(HexColor("#1e2a3a"))
        c.setLineWidth(0.5)
        c.line(div_x, y - card_h + 4 * mm, div_x, y - 4 * mm)

        # Threat verdict
        c.setFont(FONT_BRAND, 18)
        c.setFillColor(v_col)
        c.drawString(div_x + 6 * mm, y - 17 * mm, threat.upper())

        c.setFont(FONT_BODY, 7.5)
        c.setFillColor(C_TEXT_LIGHT)
        fusion = result.get("fusion_mode", "")
        c.drawString(div_x + 6 * mm, y - card_h + 8 * mm,
                     f"Fusion: {fusion}" if fusion else "Adaptive fusion")

        # Horizontal score bar inside card
        bar_y   = y - card_h + 3 * mm
        bar_x   = MARGIN_L + 8 * mm
        bar_w   = 40 * mm
        bar_h_p = 3
        c.setFillColor(HexColor("#1e2a3a"))
        c.rect(bar_x, bar_y, bar_w, bar_h_p, fill=1, stroke=0)
        c.setFillColor(v_col)
        c.rect(bar_x, bar_y, bar_w * (score / 100), bar_h_p, fill=1, stroke=0)

        y -= card_h + 6 * mm

        # ── Stage scores breakdown ────────────────────────────────────────────
        c.setFont(FONT_HEADING, 8)
        c.setFillColor(C_TEXT_DARK)
        c.drawString(MARGIN_L, y, "Stage Score Breakdown")
        c.setStrokeColor(C_BORDER)
        c.setLineWidth(0.4)
        c.line(MARGIN_L, y - 3, MARGIN_L + CONTENT_W, y - 3)
        y -= 12

        stage_scores = result.get("stage_scores", {})
        score_pairs = [
            ("Frequency / Forensic",    result.get("forensic_score",     stage_scores.get("forensic", 0))),
            ("Manipulation Analysis",   result.get("manipulation_score", stage_scores.get("manipulation", 0))),
            ("Vehicle / Object",        result.get("vehicle_score",      stage_scores.get("vehicle", 0))),
            ("Deep Learning Classifier",result.get("dl_score",           stage_scores.get("dl", 0))),
            ("Metadata Forensics",      result.get("metadata_score",     stage_scores.get("metadata", 0))),
        ]
        bar_w_full = CONTENT_W - 4 * mm
        for label, val in score_pairs:
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 0.0
            y = _draw_score_bar(c, label, val, MARGIN_L + 2 * mm, y, bar_w_full)

        y -= 4 * mm

        # ── Notes / summary ───────────────────────────────────────────────────
        notes = result.get("notes", "")
        if notes:
            c.setFont(FONT_HEADING, 8)
            c.setFillColor(C_TEXT_DARK)
            c.drawString(MARGIN_L, y, "Analyst Notes")
            c.setStrokeColor(C_BORDER)
            c.setLineWidth(0.4)
            c.line(MARGIN_L, y - 3, MARGIN_L + CONTENT_W, y - 3)
            y -= 12
            c.setFont(FONT_BODY, 7.5)
            c.setFillColor(C_TEXT_MID)
            # simple word-wrap
            words = str(notes).split()
            line_buf, line_w = [], 0
            for w in words:
                ww = c.stringWidth(w + " ", FONT_BODY, 7.5)
                if line_w + ww > CONTENT_W and line_buf:
                    c.drawString(MARGIN_L, y, " ".join(line_buf))
                    y -= 10
                    line_buf, line_w = [w], ww
                else:
                    line_buf.append(w)
                    line_w += ww
            if line_buf:
                c.drawString(MARGIN_L, y, " ".join(line_buf))
                y -= 10

        return y

    # ── Graph grouping ────────────────────────────────────────────────────────
    def _group_graphs_by_stage(self):
        """
        Group self.graphs into ordered sections by filename prefix.
        Filenames are expected to contain keywords: freq/fft/power/phase/noise/
        edge/face/ela/prnu/patch/copy/vehicle/damage/dl/attention/fusion/meta etc.
        """
        groups = {}
        stage_map = [
            ("Stage 1 — Frequency Domain Analysis",
             ["freq", "fft", "power", "phase", "radial", "band", "channel", "noise_residual", "edge"]),
            ("Stage 2 — Face Forensic Analysis",
             ["face", "blur", "sharpness"]),
            ("Stage 3 — Manipulation Analysis",
             ["ela", "prnu", "patch", "copy", "meta"]),
            ("Stage 4 — Vehicle & Object Damage Analysis",
             ["vehicle", "damage", "shadow", "texture", "boundary", "insurance"]),
            ("Stage 5 — Deep Learning Detector",
             ["dl", "attention", "patch_var", "fusion", "heatmap"]),
        ]
        unclaimed = list(self.graphs)
        for stage_label, keywords in stage_map:
            matched = []
            remaining = []
            for item in unclaimed:
                path, caption = item
                name_lower = os.path.basename(path).lower()
                if any(kw in name_lower for kw in keywords):
                    matched.append(item)
                else:
                    remaining.append(item)
            if matched:
                groups[stage_label] = matched
            unclaimed = remaining
        if unclaimed:
            groups["Additional Analysis"] = unclaimed
        return groups

    # ── Graph section pages ───────────────────────────────────────────────────
    def _draw_graph_section(self, c: canvas.Canvas, stage_label: str,
                            items: list, page_num: int, total_pages: int) -> int:
        """
        Lay out graphs for one stage across however many pages needed.
        Returns updated page_num.
        """
        # 2-column grid, max 2 rows per page = 4 graphs per page
        COLS   = 2
        G_W    = (CONTENT_W - 4 * mm) / 2      # graph width
        G_H    = G_W * 0.72                     # 4:3-ish
        GAP    = 4 * mm
        HEADING_OFFSET = 13 * mm + 4 * mm       # heading height + gap

        content_top = PAGE_H - MARGIN_T - HEADER_H - 6 * mm
        row_h       = G_H + GAP + 8             # graph + caption space

        # How many rows fit on first page (after heading) and subsequent pages
        avail_first = content_top - HEADING_OFFSET - (FOOTER_H + MARGIN_B + 4 * mm)
        avail_rest  = content_top - (FOOTER_H + MARGIN_B + 4 * mm)
        rows_first  = max(1, int(avail_first / row_h))
        rows_rest   = max(1, int(avail_rest  / row_h))

        graphs_per_first = rows_first * COLS
        graphs_per_rest  = rows_rest  * COLS

        i = 0
        first = True
        while i < len(items):
            _draw_watermark(c)
            _draw_header(c, self.job_id, page_num, total_pages)
            _draw_footer(c, self.filename, self.generated_at)

            y = content_top
            if first:
                parts = stage_label.split("—", 1) if "—" in stage_label else stage_label.split("--", 1)
                num_part   = parts[0].strip().replace("Stage", "").strip() if len(parts) > 1 else "·"
                title_part = parts[1].strip() if len(parts) > 1 else stage_label
                y = _draw_stage_heading(c, y, num_part, title_part)
                batch = graphs_per_first
                first = False
            else:
                # continuation header
                c.setFont(FONT_BODY, 7)
                c.setFillColor(C_TEXT_LIGHT)
                c.drawString(MARGIN_L, y, f"{stage_label}  (continued)")
                y -= 10
                batch = graphs_per_rest

            chunk = items[i:i + batch]
            for j, (path, caption) in enumerate(chunk):
                col = j % COLS
                row = j // COLS
                gx = MARGIN_L + col * (G_W + GAP)
                gy = y - row * row_h
                _place_graph(c, path, gx, gy, G_W, G_H, caption)

            c.showPage()
            page_num += 1
            i += batch

        return page_num

    # ── Final Verdict page ────────────────────────────────────────────────────
    def _draw_verdict_page(self, c: canvas.Canvas, result: dict):
        score   = float(result.get("score", 0))
        threat  = result.get("threat", "UNKNOWN")
        v_col   = _verdict_color(threat)

        content_top = PAGE_H - MARGIN_T - HEADER_H - 10 * mm

        # ── "FINAL VERDICT" label ─────────────────────────────────────────────
        c.setFont(FONT_BODY, 8)
        c.setFillColor(C_TEXT_LIGHT)
        c.drawCentredString(PAGE_W / 2, content_top, "FINAL VERDICT")

        # Teal top rule
        c.setStrokeColor(C_TEAL)
        c.setLineWidth(1.5)
        c.line(MARGIN_L, content_top - 4, PAGE_W - MARGIN_R, content_top - 4)

        y = content_top - 20

        # ── Big score circle ──────────────────────────────────────────────────
        cx   = PAGE_W / 2
        cy   = y - 38 * mm
        r_out = 28 * mm
        r_in  = 22 * mm

        # Outer ring (dark)
        c.setFillColor(C_DARK)
        c.circle(cx, cy, r_out, fill=1, stroke=0)

        # Arc fill for score (approximate with a pie slice drawn on top)
        # Inner circle (white punch)
        c.setFillColor(C_WHITE)
        c.circle(cx, cy, r_in, fill=1, stroke=0)

        # Score number inside
        c.setFont(FONT_BRAND, 28)
        c.setFillColor(v_col)
        score_str = f"{score:.1f}%"
        c.drawCentredString(cx, cy - 5, score_str)

        c.setFont(FONT_BODY, 6)
        c.setFillColor(C_TEXT_MID)
        c.drawCentredString(cx, cy - 16, "MANIPULATION PROBABILITY")

        # Threat level text below circle
        y_after_circle = cy - r_out - 8 * mm

        c.setFont(FONT_BRAND, 26)
        c.setFillColor(v_col)
        c.drawCentredString(cx, y_after_circle, threat.upper())

        c.setFont(FONT_BODY, 8)
        c.setFillColor(C_TEXT_MID)
        c.drawCentredString(cx, y_after_circle - 14, "Threat Level")

        y = y_after_circle - 28

        # ── Score breakdown table ─────────────────────────────────────────────
        # Centred 2-col layout
        table_w = 100 * mm
        table_x = (PAGE_W - table_w) / 2
        col_label_w = 60 * mm
        col_val_w   = table_w - col_label_w

        rows = [
            ("Forensic Analysis",          result.get("forensic_score", "—")),
            ("Manipulation Analysis",       result.get("manipulation_score", "—")),
            ("Vehicle / Object Damage",     result.get("vehicle_score", "—")),
            ("Deep Learning Classifier",    result.get("dl_score", "—")),
            ("Metadata Forensics",          result.get("metadata_score", "—")),
            ("Fusion Mode",                 result.get("fusion_mode", "—")),
        ]

        # Header row
        c.setFillColor(C_DARK)
        c.rect(table_x, y - 8, table_w, 9, fill=1, stroke=0)
        c.setFont(FONT_BRAND, 6.5)
        c.setFillColor(C_WHITE)
        c.drawString(table_x + 3, y - 5.5, "Analysis Stage")
        c.drawRightString(table_x + table_w - 3, y - 5.5, "Score")
        y -= 8

        for k, (label, val) in enumerate(rows):
            row_fill = C_OFF_WHITE if k % 2 == 0 else C_WHITE
            c.setFillColor(row_fill)
            c.setStrokeColor(C_BORDER)
            c.setLineWidth(0.3)
            c.rect(table_x, y - 8, table_w, 8, fill=1, stroke=1)
            c.setFont(FONT_BODY, 6.5)
            c.setFillColor(C_TEXT_DARK)
            c.drawString(table_x + 3, y - 5.5, label)
            try:
                num = float(val)
                display = f"{num:.1f}%"
                col = C_VERDICT_LOW if num < 40 else (C_VERDICT_MED if num < 65 else C_VERDICT_HIGH)
            except (TypeError, ValueError):
                display = str(val)
                col = C_TEXT_DARK
            c.setFillColor(col)
            c.setFont(FONT_BRAND if display.endswith("%") else FONT_BODY, 6.5)
            c.drawRightString(table_x + table_w - 3, y - 5.5, display)
            y -= 8

        y -= 10

        # ── Disclaimer ────────────────────────────────────────────────────────
        disclaimer = (
            "This report was generated automatically by the Darpan Forensic Engine v5 (AlgorivX.AI). "
            "It is intended for use by qualified forensic analysts, security researchers, journalists, "
            "and insurance investigators. Results should be interpreted alongside domain expertise. "
            "Pure colour grading and liquify/warp operations without structural change may not be detected "
            "by any current forensic tool."
        )
        c.setFont(FONT_BODY, 6)
        c.setFillColor(C_TEXT_LIGHT)
        words = disclaimer.split()
        line_buf, line_w = [], 0
        max_w = CONTENT_W - 20 * mm
        start_x = MARGIN_L + 10 * mm
        for w in words:
            ww = c.stringWidth(w + " ", FONT_BODY, 6)
            if line_w + ww > max_w and line_buf:
                c.drawCentredString(PAGE_W / 2, y, " ".join(line_buf))
                y -= 8
                line_buf, line_w = [w], ww
            else:
                line_buf.append(w)
                line_w += ww
        if line_buf:
            c.drawCentredString(PAGE_W / 2, y, " ".join(line_buf))

        # Bottom signature
        y -= 16
        c.setStrokeColor(C_TEAL)
        c.setLineWidth(0.5)
        sig_w = 60 * mm
        c.line(PAGE_W / 2 - sig_w / 2, y, PAGE_W / 2 + sig_w / 2, y)
        y -= 8
        c.setFont(FONT_BODY, 6)
        c.setFillColor(C_TEXT_LIGHT)
        c.drawCentredString(PAGE_W / 2, y, f"AlgorivX.AI · Darpan · {self.generated_at}")

    # ── Page count estimator ──────────────────────────────────────────────────
    def _count_pages(self) -> int:
        """Rough estimate: 1 cover + graph pages + 1 verdict."""
        n_graphs = len(self.graphs)
        # ~4 graphs per page
        graph_pages = max(1, math.ceil(n_graphs / 4))
        return 1 + graph_pages + 1
