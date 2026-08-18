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


def _confidence_category(classification: str, threat: str, editing_detected: bool = False) -> str:
    """
    Plain-language category label to sit above the raw threat word on the
    PDF cover, mirroring the same borderline/confident distinction added to
    page.js's VerdictHero (9 Aug 2026) - a MODERATE-risk AI_GENERATED result
    previously looked structurally identical to a CRITICAL one on paper,
    same layout, same font sizes, only the word and color differed. This
    doesn't replace the threat word or its color-coding (both stay exactly
    as before); it adds one short line of context above it.
    """
    t = (threat or '').upper()
    if classification == 'REAL':
        return 'Likely real, edited' if editing_detected else 'Likely real'
    if classification in ('AI_GENERATED', 'DEEPFAKE'):
        if t in ('HIGH', 'CRITICAL'):
            return 'Strong AI-generation signal' if classification == 'AI_GENERATED' else 'Strong deepfake signal'
        return 'Needs review — signal is weak'
    return 'Inconclusive'


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


# -- Stage breakdown whitelist ---------------------------------------------
# Mirrors frontend STAGE_DISPLAY exactly (page.js) so the PDF's "Stage Score
# Breakdown" (cover page) and "Analysis Stage Score" (verdict page) tables
# never show a different set of fields than the on-screen stage tiles.
# Previously both tables did `for k, v in stage_scores.items()` with no
# filter at all, which pulled in fields that were never meant to be
# displayed as a 0-100% score:
#   - exif_ai_corroborated is a bool (True/False), not a score -> rendered
#     as "1.0%" when True.
#   - noise_residual_std is a raw pixel-value std-dev (e.g. 2.87), not a
#     0-100 scale -> rendered as "2.9%" which looks like a score but isn't.
# Restricting to this whitelist, in this order, fixes both without needing
# any special-casing of individual keys.
STAGE_ORDER = [
    'exif_edit_score', 'frequency', 'face_forensics', 'manipulation',
    'vehicle_damage', 'deep_learning', 'dl_ai_generated',
    'vehicle_domain_ai_gen',
]
STAGE_LABELS = {
    'exif_edit_score':        'EXIF Edit Signal',
    'frequency':               'Frequency Analysis',
    'face_forensics':          'Face Forensics',
    'manipulation':            'Manipulation',
    'vehicle_damage':          'Vehicle Damage',
    'deep_learning':           'DL Deepfake',
    'dl_ai_generated':         'DL AI-Generated',
    # Absent from stage_scores (None) until vehicle_ai_gen_classifier.py
    # has an actual trained model - _stage_pairs() already skips None
    # values, so this row simply doesn't appear until then.
    'vehicle_domain_ai_gen':   'Vehicle-Domain AI-Gen',
}

def _stage_pairs(stage_scores):
    """Whitelisted, ordered (label, value) pairs for stage-score tables."""
    if not stage_scores:
        return []
    pairs = []
    for key in STAGE_ORDER:
        v = stage_scores.get(key)
        if v is not None:
            pairs.append((STAGE_LABELS[key], v))
    return pairs


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


# -- Detailed Metrics: category mapping + plain-language summaries ---------------
STAT_CATEGORIES = [
    ("Image Pattern Analysis",  "\u25A6",
     ["band energy", "hf std", "spectral entropy", "noise residual",
      "edge density", "frequency score", "frequency suspicion"],
     "frequency", "frequency patterns consistent with AI generation"),
    ("Face Analysis",            "\u25C9",
     ["face forensic", "human faces", "raw detections"],
     "face_forensics", "face-swap or synthetic-face indicators"),
    ("Editing Traces",           "\u2735",
     ["ela ", "ela score", "copy-move", "noise cv", "noise range",
      "patch outlier", "patch score", "metadata score",
      "manipulation score", "prnu"],
     "manipulation", "cloning, splicing, or retouching"),
    ("Vehicle & Damage Analysis","\u26A0",
     ["damage", "shadow", "texture", "boundary", "insurance"],
     "vehicle_damage", "inconsistencies in the damage region"),
    ("Deep Learning Detectors",  "\u25C6",
     ["dl ai-gen", "dl deepfake", "dl score", "fusion", "forensic score"],
     "deep_learning", "AI-generation or face-swap signatures"),
    ("Photo Metadata (EXIF)",    "\u2387",
     ["exif", "gps", "timestamp", "device in exif", "software tag", "camera make"],
     "exif_ai_score", "metadata inconsistent with a genuine camera photo"),
    ("File Information",         "\u25A3",
     ["format", "dimensions", "megapixels", "downscaled"],
     None, None),
]

def _categorize_stats(stats, stage_scores):
    buckets = {title: [] for title, *_ in STAT_CATEGORIES}
    unclaimed = []
    for stat in stats:
        lc = stat['label'].lower()
        placed = False
        for title, icon, kws, *_ in STAT_CATEGORIES:
            if any(kw in lc for kw in kws):
                buckets[title].append(stat); placed = True; break
        if not placed: unclaimed.append(stat)
    result = []
    for title, icon, _, stage_key, subject in STAT_CATEGORIES:
        items = buckets[title]
        if not items: continue
        summary = _stat_summary(title, stage_key, subject, stage_scores, items)
        result.append((title, icon, summary, items))
    if unclaimed: result.append(("Other", "\u25CB", None, unclaimed))
    return result

def _stat_summary(title, stage_key, subject, stage_scores, items):
    if title == "File Information":
        return "Technical file details — included for reference, not a forensic signal."
    if title == "Face Analysis":
        if any("no human face" in str(s.get('value','')).lower() for s in items):
            return "No human face was detected in this image, so face-swap checks were not applicable."
    score = stage_scores.get(stage_key) if stage_key else None
    if score is None: return None
    try: score = float(score)
    except (TypeError, ValueError): return None
    if score < 40:  return f"Low evidence of {subject} ({score:.0f}%)."
    if score < 65:  return f"Some evidence of {subject} ({score:.0f}%) — worth a closer look."
    return             f"Strong evidence of {subject} ({score:.0f}%)."


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
        self.payload      = {
            # exif_ai_corroborated defaults to explicit False rather than being
            # absent-by-default. Previously it was only ever added to
            # stage_scores when the no-EXIF corroboration block in
            # image_pipeline.py set it True (the block right after
            # dl_detector() in analyze_image()) - it was never written as
            # False anywhere, so "absent" and "False" were indistinguishable
            # in the payload/report JSON. That ambiguity is what made a
            # 4 Aug 2026 anomaly (exif_ai_score=70.0 with the key missing
            # entirely) impossible to read conclusively from the JSON alone.
            # This fix does NOT explain that anomaly - it only ensures it
            # can't happen again: the key is now always present, so its
            # exact value (not its presence) is the signal going forward.
            "job_id": job_id, "graphs": [], "indicators": [], "stats": [],
            "stage_scores": {"exif_ai_corroborated": False},
            # New REAL / AI_GENERATED / DEEPFAKE taxonomy fields (additive, Phase 1).
            "classification":        "UNKNOWN",
            "real_score":            0.0,
            "ai_generated_score":    0.0,
            "deepfake_score":        0.0,
            "confidence":            0.0,
            "confidence_level":      "LOW",
            "risk_level":            "LOW",
            "evidence":              [],
            "analysis_version":      None,
            "detectors_used":        [],
            "successful_detectors":  [],
            "failed_detectors":      [],
            "processing_time_ms":    None,
        }
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
        Saves the CURRENTLY ACTIVE matplotlib figure to disk and registers it.

        Every pipeline call site follows the same pattern:
            fig, ax = plt.subplots(...)
            ...plotting...
            R.save_graph('name.png', 'Title', 'Description', important=True)
            plt.close(fig)
        So at the moment this runs, plt.gcf() is still that exact figure -
        nothing has created a new one or closed it yet. This is what was
        missing before: save_graph only registered a filename string and
        never actually wrote any PNG bytes to disk, which is why every graph
        404'd in the frontend and showed "[graph unavailable]" in the PDF.

        Signature matches all existing pipeline calls:
            R.save_graph('name.png', 'Title', 'Description', important=True)
        """
        import matplotlib.pyplot as plt
        path = os.path.join(self.graph_dir, filename)
        fig  = plt.gcf()
        fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
        self._graphs.append((path, title))
        if important:
            self.payload["graphs"].append({
                "title":       title,
                "filename":    filename,
                "description": description,
            })
        return path

    def _draw_detailed_metrics(self, c, categorized, page_num, total_pages, filename):
        """
        Render the grouped Detailed Metrics section as a tile grid, matching
        the on-screen stat card layout (label on top, bold value below)
        instead of a flat single-column label/value list. The previous flat
        rows read as an undifferentiated wall of text; this mirrors what the
        frontend already renders as individual stat cards.
        """
        C_TILE_BG   = HexColor('#0f2035')
        C_TILE_BRD  = HexColor('#234268')
        C_CAT_BG    = HexColor('#0f2035')
        C_SUMM      = HexColor('#7ecfcf')
        C_LABEL     = HexColor('#8da3c2')
        C_VALUE     = HexColor('#e6edf3')
        PAGE_H      = A4[1]
        USABLE_TOP  = PAGE_H - 38 * mm
        USABLE_BOT  = 22 * mm

        COLS      = 3
        GRID_W    = A4[0] - MARGIN_L - MARGIN_R
        GAP       = 3 * mm
        TILE_W    = (GRID_W - (COLS - 1) * GAP) / COLS
        TILE_H    = 15 * mm

        def new_page():
            nonlocal page_num
            _draw_watermark(c)
            _draw_header(c, self.job_id, page_num, total_pages)
            _draw_footer(c, filename, self._generated_at)
            c.setFillColor(HexColor('#00d4d4'))
            c.setFont(FONT_B, 10)
            c.drawString(MARGIN_L, PAGE_H - 30 * mm, 'DETAILED METRICS')
            c.setStrokeColor(HexColor('#00d4d4'))
            c.setLineWidth(0.5)
            c.line(MARGIN_L, PAGE_H - 31.5 * mm, A4[0] - MARGIN_R, PAGE_H - 31.5 * mm)
            page_num += 1
            return USABLE_TOP

        def draw_tile(x, y_top, stat):
            """Single stat card: small muted label, bold value below."""
            c.setFillColor(C_TILE_BG)
            c.setStrokeColor(C_TILE_BRD)
            c.setLineWidth(0.5)
            c.roundRect(x, y_top - TILE_H, TILE_W, TILE_H, 2, fill=1, stroke=1)

            c.setFont(FONT_R, 6.5)
            c.setFillColor(C_LABEL)
            label = stat['label']
            max_label_w = TILE_W - 4 * mm
            if c.stringWidth(label, FONT_R, 6.5) > max_label_w:
                while label and c.stringWidth(label + '…', FONT_R, 6.5) > max_label_w:
                    label = label[:-1]
                if label != stat['label']:
                    label += '…'
            c.drawString(x + 2.5 * mm, y_top - 5.5 * mm, label)

            c.setFont(FONT_B, 10)
            c.setFillColor(C_VALUE)
            value = str(stat['value'])
            max_val_w = TILE_W - 4 * mm
            if c.stringWidth(value, FONT_B, 10) > max_val_w:
                while value and c.stringWidth(value + '…', FONT_B, 10) > max_val_w:
                    value = value[:-1]
                if value != str(stat['value']):
                    value += '…'
            c.drawString(x + 2.5 * mm, y_top - 11.5 * mm, value)

        y = new_page()

        for title, icon, summary, items in categorized:
            rows_needed = math.ceil(len(items) / COLS)
            grid_h = rows_needed * (TILE_H + GAP)
            need_h = 9*mm + (5*mm if summary else 0) + grid_h + 4*mm
            if y - need_h < USABLE_BOT:
                c.showPage()
                y = new_page()

            c.setFillColor(C_CAT_BG)
            c.rect(MARGIN_L, y - 8*mm, GRID_W, 8*mm, fill=1, stroke=0)
            c.setFillColor(HexColor('#00d4d4'))
            c.setFont(FONT_B, 8)
            c.drawString(MARGIN_L + 3*mm, y - 5.5*mm, f'{icon}  {title.upper()}')
            y -= 9*mm

            if summary:
                c.setFont(FONT_R, 7.5)
                c.setFillColor(C_SUMM)
                words = summary.split()
                line = ''
                for word in words:
                    if len(line) + len(word) + 1 > 90:
                        c.drawString(MARGIN_L + 2*mm, y, line.strip())
                        y -= 4.5*mm
                        line = word + ' '
                    else:
                        line += word + ' '
                if line.strip():
                    c.drawString(MARGIN_L + 2*mm, y, line.strip())
                    y -= 5.5*mm

            for i, stat in enumerate(items):
                col = i % COLS
                if col == 0 and y - TILE_H < USABLE_BOT:
                    c.showPage()
                    y = new_page()
                tx = MARGIN_L + col * (TILE_W + GAP)
                draw_tile(tx, y, stat)
                if col == COLS - 1 or i == len(items) - 1:
                    y -= TILE_H + GAP

            y -= 4*mm

        c.showPage()
        return page_num

    def build_pdf(self):
        """Build the professional PDF and write it to self.report_path."""
        filename    = self.payload.get("filename", "unknown")
        file_type   = self.payload.get("file_type", "")
        score          = float(self.payload.get("final_score", 0))
        threat         = self.payload.get("threat_level", "UNKNOWN")
        verdict        = self.payload.get("verdict", "")
        stage_scores   = self.payload.get("stage_scores", {})
        classification = self.payload.get("classification", "UNKNOWN")
        real_score     = float(self.payload.get("real_score", 0))
        # BUG FIX (18 Aug 2026): the screen (page.js VerdictHero) was
        # redesigned to show ONE label + a distance-from-threshold
        # "confidence" number (helpers.py classify_dominant's 'confidence'
        # field) instead of the raw score, specifically because a raw score
        # right next to a confident-sounding label was misleading at
        # borderline results. The PDF was never updated to match - it kept
        # headlining the raw score (e.g. "61.2% AUTHENTICITY SCORE") with no
        # mention of confidence at all, so a report showing "Confidence: 57%"
        # on screen and "61.2%" as the PDF's big number look like two
        # different, disagreeing results, even though both are computed from
        # the exact same payload. Confirmed directly: report 7c6d944bf22b
        # showed Confidence 57% on screen vs 61.2% Authenticity Score in the
        # PDF - correct math on both sides, but no shared anchor between
        # them. Reads it here with the same default (50.0, "exactly at the
        # threshold") verdict_text_v2 and page.js already use for payloads
        # that predate this field.
        confidence     = float(self.payload.get("confidence", 50.0))
        # NEW (4 Aug 2026, part of the same display_score fix below): needed
        # so the header LABEL can also distinguish REAL (Edited) from plain
        # REAL, not just the number. See _draw_cover/_draw_verdict_page.
        editing_detected = bool(self.payload.get("editing_detected", False))
        stats          = self.payload.get("stats", [])

        # display_score is what actually gets headlined as the big number.
        # Previously the cover/verdict pages always headlined `score`
        # (= final_score, the synthetic-signal strength) even when the
        # label above it said "AUTHENTICITY SCORE" for REAL results - e.g.
        # a REAL result showing "39.9% / AUTHENTICITY SCORE" with a small
        # footnote clarifying the real authenticity number was actually
        # 60.1%. The footnote was a patch over a label that didn't match
        # the number. Now the headline and label always agree; `score`
        # (raw synthetic strength) is kept as a secondary line for REAL
        # results, since it's still useful context (matches Stage Score
        # Breakdown / Final Score row), just no longer the misleading
        # headline.
        #
        # BUG FIX (4 Aug 2026): the line above only branched on
        # `classification == "REAL"`, not `editing_detected`. For a
        # REAL (Edited) result, classify_dominant() sets dominant_score =
        # edit_composite (editing-evidence strength) and that's exactly
        # what app/page.js's VerdictHero headlines via result.dominant_score
        # - but this line still fell back to real_score (~100 - synthetic
        # signal, unrelated to editing evidence) for every REAL result
        # regardless of editing_detected. Same payload, two different
        # headline numbers on screen vs PDF for REAL (Edited) results.
        # helpers.py's own classify_dominant() docstring already documents
        # payload['dominant_score'] as "what the frontend shows as the
        # headline number" - so read that field directly instead of
        # re-deriving a second, now-stale copy of the same decision here.
        # Falls back to the old formula only for payloads from before
        # dominant_score existed.
        display_score = float(self.payload.get(
            "dominant_score",
            real_score if classification == "REAL" else score
        ))

        # FIX (9 Aug 2026): total_pages was previously a static upfront
        # estimate (_estimate_pages(), still below for reference/removal
        # candidate) that never accounted for the detailed-metrics pages
        # _draw_detailed_metrics() adds - confirmed by reproducing the
        # "Page 7 of 6" / "8 of 6" / "9 of 6" bug directly: page_num kept
        # incrementing correctly as pages were actually drawn, but the "of
        # N" denominator baked into every header was fixed from the start
        # and never caught up. Root-cause fix, not a better estimate: run
        # the exact same page-generation logic once against a throwaway
        # canvas purely to find the true final page_num, then run it again
        # for the real output using that as the correct total. Doubles PDF
        # generation time but that's a non-issue next to model inference
        # time elsewhere in this pipeline, and it's the only way to get an
        # accurate denominator for content whose page count depends on
        # how many stats/graphs a given image actually produced.
        dry_buf = io.BytesIO()
        dry_c   = canvas.Canvas(dry_buf, pagesize=A4)
        true_total_pages = self._render_pages(
            dry_c, total_pages=1, filename=filename, file_type=file_type,
            display_score=display_score, threat=threat, verdict=verdict,
            stage_scores=stage_scores, classification=classification,
            score=score, editing_detected=editing_detected, stats=stats,
            confidence=confidence,
        )

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        c.setTitle(f"Darpan Forensic Report  -  {self.job_id}")
        c.setAuthor("AlgorivX.AI . Darpan Forensic Engine v5")
        self._render_pages(
            c, total_pages=true_total_pages, filename=filename, file_type=file_type,
            display_score=display_score, threat=threat, verdict=verdict,
            stage_scores=stage_scores, classification=classification,
            score=score, editing_detected=editing_detected, stats=stats,
            confidence=confidence,
        )

        c.save()
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        with open(self.report_path, "wb") as f:
            f.write(buf.getvalue())

    def _render_pages(self, c, total_pages, filename, file_type, display_score,
                       threat, verdict, stage_scores, classification, score,
                       editing_detected, stats, confidence=50.0):
        """
        Draws every page of the report onto canvas c, using total_pages as
        the header denominator throughout. Returns the true final page_num
        reached - used by build_pdf() as the dry-run pass to discover the
        correct total_pages for the real pass. Pulled out of build_pdf()
        specifically so both passes run the exact same logic - a second,
        separately-maintained "count pages" function would drift from the
        real drawing code the moment either one changed.
        """
        page_num = 1

        # -- PAGE 1: Cover + Summary -------------------------------------------
        _draw_watermark(c)
        _draw_header(c, self.job_id, page_num, total_pages)
        _draw_footer(c, filename, self._generated_at)
        self._draw_cover(c, display_score, threat, verdict, stage_scores, file_type, filename, classification, score, editing_detected, confidence)
        c.showPage()
        page_num += 1

        # -- Graph pages by stage ----------------------------------------------
        groups = self._group_graphs()
        for stage_label, items in groups.items():
            page_num = self._draw_graph_section(c, stage_label, items, page_num, total_pages)

        # -- Detailed Metrics section (grouped stats, plain-language summaries) --
        if stats:
            categorized = _categorize_stats(stats, stage_scores)
            if categorized:
                page_num = self._draw_detailed_metrics(
                    c, categorized, page_num, total_pages, filename
                )

        # -- Final Verdict page ------------------------------------------------
        _draw_watermark(c)
        _draw_header(c, self.job_id, page_num, total_pages)
        _draw_footer(c, filename, self._generated_at)
        self._draw_verdict_page(c, display_score, threat, stage_scores, verdict, classification, score, editing_detected)
        c.showPage()

        return page_num

    # -- Internal PDF builders -------------------------------------------------

    def _draw_cover(self, c, score, threat, verdict, stage_scores, file_type, filename, classification='UNKNOWN', raw_score=0.0, editing_detected=False, confidence=50.0):
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

        # BUG FIX (18 Aug 2026): screen shows CLASSIFICATION + a distance-
        # from-threshold "confidence" number as its only headline (see
        # build_pdf's confidence comment above) - this PDF number is a
        # different, raw-score concept, correct on its own terms but with
        # no visible anchor back to what the reader saw on screen. Drawn
        # here as its own clearly-labeled line, matching the screen exactly
        # (same 'confidence' field, same classification label) - not
        # replacing the raw score below, which stays as legitimate detail,
        # but giving the reader an immediate "yes, this is the same report"
        # checkpoint before the raw technical numbers. Placed in the
        # previously-unused header space at the top of the score card - the
        # 44pt score number's baseline sits at y-25mm with real vertical
        # extent both above and below that, so this must NOT share that
        # band; y-6mm is comfortably clear of it.
        classification_display = {
            'REAL': 'REAL', 'AI_GENERATED': 'AI GENERATED', 'DEEPFAKE': 'DEEPFAKE',
        }.get(classification, classification)
        if classification == "REAL" and editing_detected:
            classification_display = 'REAL (Edited)'
        c.setFont(FONT_B, 9)
        c.setFillColor(v_col)
        c.drawString(MARGIN_L + 8 * mm, y - 6 * mm,
                     f"App shows: {classification_display}  \u00b7  Confidence: {confidence:.0f}%")

        c.setFont(FONT_B, 44)
        c.setFillColor(v_col)
        c.drawString(MARGIN_L + 8 * mm, y - 25 * mm, f"{score:.1f}%")

        c.setFont(FONT_R, 7)
        c.setFillColor(C_TEXT_LIGHT)
        # LABEL FIX (4 Aug 2026, same root cause as the display_score fix
        # above): this used to check only `classification == "REAL"`, so a
        # REAL (Edited) result showed the edit-evidence number under
        # "AUTHENTICITY SCORE" - e.g. "42.0% AUTHENTICITY SCORE" reads as
        # low authenticity on a genuinely real photo, when 42.0 is actually
        # edit_composite (how strong the editing evidence is), not how real
        # the image is. Now three-way, matching the three cases
        # classify_dominant() actually produces.
        if classification == "REAL" and editing_detected:
            prob_label = "EDIT SIGNAL STRENGTH"
        elif classification == "REAL":
            prob_label = "AUTHENTICITY SCORE"
        else:
            prob_label = "MANIPULATION PROBABILITY"
        c.drawString(MARGIN_L + 8 * mm, y - card_h + 5 * mm, prob_label)

        # Secondary note only for plain REAL (not edited): for REAL (Edited),
        # raw_score (=final_score) now equals the headline number exactly
        # (both are edit_composite - see the display_score fix above), so
        # this note would just repeat the headline under a "synthetic
        # signal" framing that doesn't apply to an edited-but-real image.
        if classification == "REAL" and not editing_detected:
            c.setFont(FONT_R, 6)
            c.setFillColor(C_TEXT_LIGHT)
            note = f"Synthetic signal strength: {raw_score:.1f}% — see Stage Score Breakdown below"
            c.drawString(MARGIN_L + 8 * mm, y - card_h - 2 * mm, note)

        div_x = MARGIN_L + 52 * mm
        c.setStrokeColor(HexColor("#1e2a3a"))
        c.setLineWidth(0.5)
        c.line(div_x, y - card_h + 4 * mm, div_x, y - 4 * mm)

        c.setFont(FONT_R, 7.5)
        c.setFillColor(C_TEXT_LIGHT)
        c.drawString(div_x + 6 * mm, y - 9 * mm, _confidence_category(classification, threat, editing_detected).upper())

        c.setFont(FONT_B, 18)
        c.setFillColor(v_col)
        c.drawString(div_x + 6 * mm, y - 20 * mm, threat.upper())

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

        pairs = _stage_pairs(stage_scores)
        if not pairs:
            pairs = [("Final Score", raw_score)]

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

    def _draw_verdict_page(self, c, score, threat, stage_scores, verdict, classification='UNKNOWN', raw_score=None, editing_detected=False):
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
        # Same three-way label fix as _draw_cover - see the comment there
        # for the full rationale (4 Aug 2026).
        if classification == "REAL" and editing_detected:
            prob_label = "EDIT SIGNAL STRENGTH"
        elif classification == "REAL":
            prob_label = "AUTHENTICITY SCORE"
        else:
            prob_label = "MANIPULATION PROBABILITY"
        c.drawCentredString(cx, cy - 16, prob_label)
        # Secondary line only for plain REAL (not edited) - see _draw_cover
        # for why this would be a redundant repeat of the headline for a
        # REAL (Edited) result.
        if classification == "REAL" and not editing_detected and raw_score is not None:
            c.setFont(FONT_R, 5)
            c.setFillColor(C_TEXT_LIGHT)
            c.drawCentredString(cx, cy - 22, f"(synthetic signal strength: {raw_score:.1f}%)")

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

        rows = _stage_pairs(stage_scores)
        rows.append(("Final Score", raw_score if raw_score is not None else score))

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
