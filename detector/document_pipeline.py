"""
Document analysis pipeline — AI-generated text detection using
Hello-SimpleAI/chatgpt-detector-roberta plus linguistic statistics.
Logic carried over unchanged from the validated Colab prototype (Cell 11).
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import docx as python_docx
from transformers import pipeline as hf_pipeline

from .result import AnalysisResult
from .helpers import apply_graph_style

# ── Model singleton — loaded once, reused per request.
# Loading inline per-request caused OOM kills (same pattern as image pipeline).
_DOC_DETECTOR = None

def _get_doc_detector():
    global _DOC_DETECTOR
    if _DOC_DETECTOR is None:
        _DOC_DETECTOR = hf_pipeline(
            'text-classification',
            model='Hello-SimpleAI/chatgpt-detector-roberta'
        )
    return _DOC_DETECTOR


def analyze_document(filepath, R: AnalysisResult):
    R.pdf_text('DOCUMENT FORENSIC ANALYSIS REPORT', 'Title')
    apply_graph_style()

    ext = os.path.splitext(filepath)[1].lower()
    text_content = ''

    if ext == '.pdf':
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t: text_content += t + '\n'
    elif ext == '.docx':
        text_content = '\n'.join(p.text for p in python_docx.Document(filepath).paragraphs)
    elif ext in ('.txt', '.md', '.rtf'):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text_content = f.read()
    else:
        raise ValueError(f'Unsupported document format: {ext}')

    if not text_content.strip():
        raise ValueError('No text could be extracted from document')

    words     = re.findall(r'\w+', text_content.lower())
    sentences = re.split(r'[.!?]+', text_content)
    sent_lens = [len(s.split()) for s in sentences if s.strip()]

    vocab_diversity = len(set(words)) / max(len(words), 1)
    avg_sent_len    = float(np.mean(sent_lens)) if sent_lens else 0

    R.add_stat('Word Count',           f'{len(words):,}')
    R.add_stat('Unique Words',         f'{len(set(words)):,}')
    R.add_stat('Vocabulary Diversity', f'{vocab_diversity:.4f}')
    R.add_stat('Avg Sentence Length',  f'{avg_sent_len:.1f} words')

    # ── Graph 1: Word frequency + sentence length dist (IMPORTANT)
    common = Counter(words).most_common(20)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].bar([w for w,_ in common], [c for _,c in common], color='#58a6ff')
    axes[0].set_title('Top 20 Words', color='#c9d1d9')
    axes[0].set_ylabel('Frequency'); plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    axes[1].hist(sent_lens, bins=30, color='#3fb950', edgecolor='#0d1117')
    axes[1].axvline(x=avg_sent_len, color='#f85149', linestyle='--', label=f'Mean={avg_sent_len:.1f}')
    axes[1].set_title('Sentence Length Distribution  — AI text tends to be uniform', color='#c9d1d9')
    axes[1].set_xlabel('Words per sentence'); axes[1].legend()

    plt.suptitle('Document Linguistic Analysis', color='#58a6ff', fontsize=13, fontweight='bold')
    plt.tight_layout()
    R.save_graph('doc_linguistics.png', 'Linguistic Analysis',
                 'Word frequency and sentence length distribution. AI-generated text shows unnaturally uniform sentence lengths.', important=True)
    plt.close(fig)

    # TF-IDF
    try:
        vec   = TfidfVectorizer(stop_words='english', max_features=30)
        mat   = vec.fit_transform([text_content])
        names = vec.get_feature_names_out()
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(names, mat.toarray()[0], color='#bc8cff')
        ax.set_title('TF-IDF Key Terms', color='#c9d1d9')
        plt.xticks(rotation=90); plt.tight_layout()
        R.save_graph('doc_tfidf.png', 'TF-IDF Key Terms',
                     'Top key terms by TF-IDF score.', important=True)
        plt.close(fig)
    except Exception as e:
        R.pdf_text(f'TF-IDF error: {e}')

    # AI text detector
    label, confidence = 'UNKNOWN', 50.0
    try:
        det_pipe    = _get_doc_detector()
        chunk_size  = 500
        words_list  = text_content.split()
        chunks      = [' '.join(words_list[i:i+chunk_size]) for i in range(0, len(words_list), chunk_size)]
        scores_list = []
        for chunk in chunks[:5]:
            res = det_pipe(chunk[:1500])[0]
            if res['label'].lower() in ('chatgpt', 'ai', 'fake', 'generated'):
                scores_list.append(res['score'] * 100)
            else:
                scores_list.append((1 - res['score']) * 100)
        confidence = float(np.mean(scores_list))
        label      = 'AI-Generated' if confidence > 50 else 'Human-Written'
        R.add_stat('AI Detector Result', f'{label} ({confidence:.1f}%)')
    except Exception as e:
        R.pdf_text(f'AI text detector error: {e}')
        R.add_stat('AI Detector', 'Unavailable')

    # Entropy
    counts       = np.array(list(Counter(text_content).values()), dtype=float)
    probs        = counts / counts.sum()
    char_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    R.add_stat('Character Entropy', f'{char_entropy:.4f}')

    # ── Graph 3: Score dashboard (IMPORTANT)
    entropy_score    = max(0, 100 - char_entropy * 10)
    vocab_score      = max(0, 100 - vocab_diversity * 100)
    uniformity_score = max(0, 100 - float(np.std(sent_lens)) * 3) if sent_lens else 50
    final_score      = confidence * 0.5 + entropy_score * 0.2 + uniformity_score * 0.3

    dashboard_scores = {
        'AI Detector'    : confidence,
        'Char Entropy'   : entropy_score,
        'Vocab Diversity': vocab_score,
        'Sent Uniformity': uniformity_score,
        'Final Score'    : final_score,
    }
    fig, ax = plt.subplots(figsize=(12, 5))
    bar_colors = ['#f85149', '#d29922', '#58a6ff', '#3fb950', '#bc8cff']
    bars = ax.bar(list(dashboard_scores.keys()), list(dashboard_scores.values()), color=bar_colors, width=0.5)
    ax.axhline(y=50, color='#f85149', linestyle='--', linewidth=1.5, label='50% threshold')
    for bar, val in zip(bars, dashboard_scores.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f'{val:.0f}', ha='center', color='#c9d1d9', fontweight='bold')
    ax.set_ylim(0, 115); ax.set_ylabel('Score (0-100)')
    ax.set_title(f'Document AI Detection Dashboard — AI Label: {label}', color='#c9d1d9')
    ax.legend(); plt.tight_layout()
    R.save_graph('doc_dashboard.png', 'AI Detection Dashboard',
                 f'Combined AI detection scores. Final score: {final_score:.1f}%. Label: {label}.', important=True)
    plt.close(fig)

    R.payload['stage_scores']['ai_text_detection']  = round(confidence, 1)
    R.payload['stage_scores']['document_forensics'] = round(final_score, 1)

    # Two-track split (mirrors the image pipeline). A document has no
    # identity to impersonate - "deepfake" isn't a meaningful concept for
    # text, so document_deepfake_score is hardcoded 0, not computed.
    R.payload['stage_scores']['document_ai_generated'] = round(final_score, 1)
    R.payload['stage_scores']['document_deepfake']     = 0.0

    # Same gap found and fixed in the image pipeline: the AI-text-detector
    # result - often the strongest single signal - previously produced zero
    # indicator text, only a stat. Without this, indicator filtering would
    # show an empty "Flagged Indicators" list even on a clearly AI-written
    # document.
    if confidence >= 65:
        R.add_indicator(f'[Document] AI-text detector flagged this content ({label}, confidence={confidence:.1f}%)')
    if uniformity_score >= 60:
        R.add_indicator(f'[Document] Unusually uniform sentence lengths (std={float(np.std(sent_lens)) if sent_lens else 0:.1f}) — consistent with AI-generated text')

    return float(np.clip(final_score, 0, 100))
