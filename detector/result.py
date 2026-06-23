"""
Result collector — builds both the JSON payload returned to the frontend
and the PDF forensic report, in parallel, as analysis runs.
Logic unchanged from the validated Colab prototype (Cell 4); only the
import paths and folder config were adapted for a standalone package.
"""
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.platypus import Image as PDFImage
from reportlab.platypus import Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

from .config import TMP_FOLDER, REPORT_FOLDER


class AnalysisResult:
    """Collects all analysis data during a pipeline run.
    Produces both the JSON response and the PDF report."""

    def __init__(self, job_id: str):
        self.job_id       = job_id
        self.report_path  = os.path.join(REPORT_FOLDER, f'report_{job_id}.pdf')
        self.tmp_dir      = os.path.join(TMP_FOLDER, job_id)
        os.makedirs(self.tmp_dir, exist_ok=True)

        # PDF setup
        self.doc      = SimpleDocTemplate(self.report_path)
        self.styles   = getSampleStyleSheet()
        self.elements = []

        # JSON payload — what gets sent to frontend
        self.payload = {
            'job_id'     : job_id,
            'file_type'  : None,
            'filename'   : None,
            'timestamp'  : datetime.now().isoformat(),
            'final_score': None,   # 0-100 deepfake probability
            'threat_level': None,  # MINIMAL / LOW / MODERATE / HIGH / CRITICAL
            'verdict'    : None,   # human-readable verdict string
            'stage_scores': {},    # per-stage scores
            'indicators' : [],     # list of triggered forensic indicators
            'metadata'   : {},     # file metadata
            'graphs'     : [],     # list of {title, description, image_b64}
            'stats'      : [],     # list of {label, value} for stats cards
            'error'      : None,
        }

    # ── PDF helpers ──────────────────────────────────────────
    def pdf_text(self, text, style='BodyText'):
        self.elements.append(Paragraph(str(text), self.styles[style]))
        self.elements.append(Spacer(1, 8))

    def pdf_image(self, path, width=5, height=5):
        if os.path.exists(path):
            self.elements.append(PDFImage(path, width=width*inch, height=height*inch))
            self.elements.append(Spacer(1, 6))

    def pdf_table(self, data, col_widths=None):
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f0f0f0'), colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        self.elements.append(t)
        self.elements.append(Spacer(1, 10))

    # ── Graph helper — saves fig, adds filename ref to payload, adds to PDF ─
    def save_graph(self, filename, title, description='', width=6, height=4, important=True):
        """Save current matplotlib figure.
        If important=True → included in frontend JSON as a filename reference.
        Frontend fetches each graph via GET /graph/{job_id}/{filename} separately,
        keeping the main /analyze JSON response small (~5KB instead of ~3.6MB).
        """
        path = os.path.join(self.tmp_dir, filename)
        plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        self.pdf_image(path, width=width, height=height)
        if important:
            self.payload['graphs'].append({
                'title'      : title,
                'description': description,
                'filename'   : filename,
            })
        return path

    def add_stat(self, label, value):
        self.payload['stats'].append({'label': label, 'value': str(value)})

    def add_indicator(self, text):
        self.payload['indicators'].append(text)

    def build_pdf(self):
        # Final verdict banner in PDF
        score = self.payload['final_score'] or 0
        level = self.payload['threat_level'] or 'UNKNOWN'
        color = ('#c0392b' if score >= 75 else
                 '#e67e22' if score >= 50 else
                 '#27ae60')
        self.elements.append(Spacer(1, 20))
        banner_data = [['FINAL VERDICT', f'{score:.1f}% Deepfake Probability', f'Threat: {level}']]
        bt = Table(banner_data, colWidths=[160, 220, 150])
        bt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(color)),
            ('TEXTCOLOR',  (0,0), (-1,-1), colors.white),
            ('FONTNAME',   (0,0), (-1,-1), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 13),
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('PADDING',    (0,0), (-1,-1), 14),
        ]))
        self.elements.append(bt)
        self.doc.build(self.elements)
        return self.report_path
