#!/usr/bin/env python3
"""
Adobe India Hackathon 2025 – Challenge 1B
Persona‑Driven Multi‑Collection PDF Intelligence

This script:
  * Walks one or more Collection_* directories (or a single collection via CLI).
  * Loads that collection's challenge1b_input.json (persona + job + docs list).
  * Extracts text from PDFs (PyPDF2; offline, CPU‑safe).
  * If Challenge 1A outline JSONs are available (auto‑detected), uses them to
    anchor section extraction (title + page ranges) for much higher precision.
  * Otherwise falls back to heuristic header/regex segmentation per page.
  * Scores sections for persona/job relevance (keywords + TF‑IDF sim + heuristics).
  * Returns top sections + representative sentences (subsections).
  * Writes challenge1b_output.json in each collection directory.

Env vars:
  TECHVERSE_1A_OUTDIR      -> override path to directory of 1A outline JSONs
  TECHVERSE_MAX_SECTIONS   -> int, default 10
  TECHVERSE_MAX_SUBSECTS   -> int, default 3
  TECHVERSE_DEBUG          -> '1' for verbose logging

Constraints compliance:
  * Offline: no network calls. NLTK downloads are NOT attempted at runtime.
    Fallback tokenizers & stopwords included (tiny internal lists) if NLTK data missing.
  * CPU only; pure Python + scikit‑learn (small TF‑IDF matrix).
  * <1GB model; we train TF‑IDF per collection (lightweight).
  * Performance: Works within 60s for ~3‑5 modest PDFs on CPU2GHz class hardware.

Author: You :)
"""

import json
import os
import re
import sys
import time
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import PyPDF2

# Lightweight NLP
import nltk
from nltk.corpus import stopwords  # may not be present in container; guarded
from nltk.tokenize import sent_tokenize, word_tokenize  # guarded

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------------------------
# Debug prints
# ------------------------------------------------------------------------------
DEBUG = os.getenv("TECHVERSE_DEBUG", "0") == "1"
def dprint(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)


# ------------------------------------------------------------------------------
# Offline-safe stopwords + tokenizers
# ------------------------------------------------------------------------------

# Minimal embedded English stopwords (fallback)
_FALLBACK_STOPWORDS = {
    "a","an","the","and","or","but","if","then","while","for","to","of","in","on",
    "with","as","by","at","from","this","that","these","those","it","its","be",
    "is","are","was","were","been","will","would","can","could","should","may",
    "might","into","over","under","about","after","before","between","so","such",
    "than","too","very","not","no","do","does","did","done","have","has","had",
    "you","your","yours","we","our","ours","they","their","them"
}

def _ensure_nltk_data():
    """
    Try to verify NLTK tokenizers/stopwords are present.
    We NEVER download at runtime (offline rule).
    """
    have_punkt = True
    have_sw = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        have_punkt = False
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        have_sw = False
    return have_punkt, have_sw

_HAVE_PUNKT, _HAVE_STOPWORDS = _ensure_nltk_data()

def tokenize_sentences(text: str) -> List[str]:
    if _HAVE_PUNKT:
        try:
            return sent_tokenize(text)
        except Exception:
            pass
    # fallback: split on .?! + newline
    return [s.strip() for s in re.split(r'[.!?]\s+|\n{2,}', text) if s.strip()]

def tokenize_words(text: str) -> List[str]:
    if _HAVE_PUNKT:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    # fallback: simple split on non-letters
    return [t for t in re.split(r'[^A-Za-z]+', text) if t]

def get_stopwords() -> set:
    if _HAVE_STOPWORDS:
        try:
            return set(stopwords.words('english'))
        except Exception:
            pass
    return _FALLBACK_STOPWORDS.copy()


# ------------------------------------------------------------------------------
# 1A Outline Loading
# ------------------------------------------------------------------------------

def locate_1a_outline(pdf_stem: str, collection_dir: Path) -> Optional[Path]:
    """
    Attempt to locate the corresponding 1A outline JSON for a given PDF *stem*.
    Lookup order:
      1. Env var TECHVERSE_1A_OUTDIR/<stem>.json
      2. collection_dir / "outlines" / <stem>.json
      3. sibling Challenge_1A/output/<stem>.json (walk upward)
    Return path or None.
    """
    # Env override
    env_dir = os.getenv("TECHVERSE_1A_OUTDIR")
    if env_dir:
        p = Path(env_dir) / f"{pdf_stem}.json"
        if p.is_file():
            dprint(f"Found 1A outline via env: {p}")
            return p

    # Local outlines folder inside collection
    p = collection_dir / "outlines" / f"{pdf_stem}.json"
    if p.is_file():
        dprint(f"Found 1A outline in collection/outlines: {p}")
        return p

    # Sibling Challenge_1A/output
    # climb up from collection_dir until we see Challenge_1B or root
    cur = collection_dir
    for _ in range(4):
        if (cur / "Challenge_1A").exists():  # parent layout
            p = cur / "Challenge_1A" / "output" / f"{pdf_stem}.json"
            if p.is_file():
                dprint(f"Found 1A outline in sibling Challenge_1A/output: {p}")
                return p
        cur = cur.parent

    return None


def load_1a_outline(outline_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(outline_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        dprint(f"Failed to load 1A outline {outline_path}: {e}")
        return None


# ------------------------------------------------------------------------------
# PDF text extraction
# ------------------------------------------------------------------------------

def extract_pages_pdf(pdf_path: Path) -> List[str]:
    """
    Extract text per page. Returns list index=page-1 -> text (str).
    Uses PyPDF2; safe for offline; some PDFs may yield empty strings.
    """
    pages = []
    try:
        with pdf_path.open("rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                pages.append(txt)
    except Exception as e:
        dprint(f"Error reading PDF {pdf_path}: {e}")
    return pages


# ------------------------------------------------------------------------------
# Heuristic section segmentation (fallback when no 1A outline)
# ------------------------------------------------------------------------------

# Patterns aggregated from your original script (reduced & generalized)
_SECTION_PATTERNS = [
    r'\n\s*(ABSTRACT|INTRODUCTION|METHODOLOGY|METHODS|RESULTS|DISCUSSION|CONCLUSION|REFERENCES)\s*\n',
    r'\n\s*(ACCOMMODATION|HOTELS?|RESTAURANTS?|DINING|FOOD|ATTRACTIONS?|SIGHTS?|TRANSPORT(?:ATION)?|ACTIVITIES|ITINERARY|BUDGET|COST)\s*\n',
    r'\n\s*(CREAT(?:E|ING)|FORMS?|FIELDS?|VALIDATION|WORKFLOW|TUTORIAL|GUIDE|STEP\s*\d+)\s*\n',
    r'\n\s*(INGREDIENTS|PREPARATION|INSTRUCTIONS|DIRECTIONS|COOKING|SERV(?:E|ING)|NUTRITION|RECIPE|MENU)\s*\n',
    r'\n\s*Chapter\s+\d+\s*\n',
    r'\n\s*Section\s+\d+\s*\n',
    r'\n\s*\d+\.?\s+[A-Z][^\n]{5,80}\n',  # numbered uppercase-ish header
    r'\n\s*[A-Z][A-Z\s]{3,}\n',           # all caps line
]

_SECTION_REGEXES = [re.compile(p, re.IGNORECASE|re.MULTILINE) for p in _SECTION_PATTERNS]


def segment_text_fallback(text: str) -> List[Tuple[str,str]]:
    """
    Fallback segmenter. Returns list of (title, content).
    1. Try pattern boundaries.
    2. Fallback to paragraphs.
    """
    boundaries = []
    for rgx in _SECTION_REGEXES:
        for m in rgx.finditer(text):
            boundaries.append((m.start(), m.end(), m.group().strip()))
    boundaries.sort(key=lambda x: x[0])

    sections: List[Tuple[str,str]] = []
    if not boundaries:
        # paragraph fallback
        paras = [p.strip() for p in re.split(r'\n\s*\n+', text) if p.strip()]
        for i, para in enumerate(paras, 1):
            if len(para) > 50:
                sections.append((f"Section {i}", para))
        return sections

    for i, (s, e, title) in enumerate(boundaries):
        start = e
        end = boundaries[i+1][0] if i+1 < len(boundaries) else len(text)
        body = text[start:end].strip()
        if len(body) > 50:
            sections.append((title, body))
    return sections


# ------------------------------------------------------------------------------
# Section building using 1A outline
# ------------------------------------------------------------------------------

@dataclass
class Section:
    document: str
    section_title: str
    start_page: int
    end_page: int
    content: str
    relevance: float = 0.0  # filled later


def build_sections_from_outline(pdf_stem: str,
                                pages: List[str],
                                outline: Dict[str, Any]) -> List[Section]:
    """
    Build sections by anchoring to 1A outline page numbers.
    Each heading starts a section from heading.page until (next.heading.page - 1).
    If multiple headings on same page we'll still segment by heading order
    but page range remains same; we don't slice page text by coordinates
    (we don't have them) so all same-page headings share same content unless
    we deduplicate titles (we dedupe later).
    """
    items = outline.get("outline", [])
    if not items:
        return []

    # sort by page (stable)
    items_sorted = sorted(items, key=lambda h: (h.get("page", 1), h.get("level","H1")))
    sections: List[Section] = []
    for idx, h in enumerate(items_sorted):
        title = h.get("text","").strip() or f"Untitled {idx+1}"
        start_pg = max(1, int(h.get("page", 1)))
        # next heading page minus 1
        if idx + 1 < len(items_sorted):
            next_pg = max(1, int(items_sorted[idx+1].get("page", start_pg)))
            end_pg = max(start_pg, next_pg - 1)
        else:
            end_pg = len(pages)
        # slice page texts
        slice_txt = "\n".join(pages[start_pg-1:end_pg])  # pages are 0-index
        sections.append(Section(
            document=f"{pdf_stem}.pdf",
            section_title=title,
            start_page=start_pg,
            end_page=end_pg,
            content=slice_txt
        ))
    # optional dedupe: combine adjacent sections with tiny content
    combined: List[Section] = []
    for sec in sections:
        if len(sec.content.strip()) < 40 and combined:
            # merge into previous
            prev = combined[-1]
            prev.content += f"\n{sec.section_title}\n{sec.content}"
            prev.end_page = max(prev.end_page, sec.end_page)
        else:
            combined.append(sec)
    return combined


# ------------------------------------------------------------------------------
# Persona / Job keyword extraction
# ------------------------------------------------------------------------------

# Domain lexicons (very small; extend if desired)
TRAVEL_TERMS = {
    "travel","trip","hotel","accommodation","stay","hostel","airbnb","restaurant",
    "food","dining","wine","beach","museum","tour","sightseeing","itinerary",
    "transport","train","bus","car","rental","budget","book","booking","city","village"
}
HR_ACROBAT_TERMS = {
    "form","fillable","pdf","acrobat","onboarding","compliance","hr","employee",
    "workflow","digital","signature","field","checkbox","dropdown","validation",
    "template","automation","submit","approval"
}
FOOD_TERMS = {
    "recipe","cook","cooking","vegetarian","vegan","buffet","menu","dish","ingredient",
    "prep","preparation","bake","boil","saute","serve","serving","portion","dietary",
    "nutrition","appetizer","main","dessert","soup","salad"
}


def persona_job_keywords(persona: str, job: str) -> List[str]:
    txt = f"{persona} {job}".lower()
    words = set(re.findall(r"[a-z]{3,}", txt))

    # domain expansion
    if any(t in txt for t in ("travel","trip","tour","planner","vacation","itinerary")):
        words |= TRAVEL_TERMS
    if any(t in txt for t in ("hr","human","form","acrobat","pdf","employee","onboard")):
        words |= HR_ACROBAT_TERMS
    if any(t in txt for t in ("cook","recipe","menu","buffet","vegetarian","food","cater")):
        words |= FOOD_TERMS

    return sorted(words)


# ------------------------------------------------------------------------------
# Text preprocessing
# ------------------------------------------------------------------------------

STOPWORDS = get_stopwords()

_WORD_RE = re.compile(r"[A-Za-z]{2,}")

def preprocess(text: str) -> str:
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    toks = [t for t in toks if t not in STOPWORDS]
    return " ".join(toks)


# ------------------------------------------------------------------------------
# Relevance Scoring
# ------------------------------------------------------------------------------

def score_section(sec_text: str,
                  persona: str,
                  job: str,
                  kw_cache: Optional[List[str]]=None,
                  tfidf_vectorizer: Optional[TfidfVectorizer]=None,
                  query_vec=None) -> float:
    """
    Combine:
      * TF-IDF cosine to persona+job query
      * Keyword coverage
      * Length min
    Expect pre-fit tfidf_vectorizer & query_vec for efficiency.
    """
    if not kw_cache:
        kw_cache = persona_job_keywords(persona, job)
    kw_set = set(kw_cache)

    # TF-IDF sim
    if tfidf_vectorizer is not None and query_vec is not None:
        try:
            sec_vec = tfidf_vectorizer.transform([preprocess(sec_text)])
            sim = float(cosine_similarity(sec_vec, query_vec)[0][0])
        except Exception:
            sim = 0.0
    else:
        sim = 0.0

    # keyword coverage (raw)
    lower = sec_text.lower()
    hits = sum(1 for k in kw_set if k in lower)
    kw_score = hits / max(len(kw_set), 1)

    # length bonus
    length_score = min(len(sec_text) / 1500.0, 1.0)

    # Weighted sum (tune as needed)
    return 0.6*sim + 0.3*kw_score + 0.1*length_score


# ------------------------------------------------------------------------------
# Subsection extraction (top sentences)
# ------------------------------------------------------------------------------

def pick_top_sentences(text: str, max_n: int = 3) -> List[str]:
    sents = tokenize_sentences(text)
    if not sents:
        return []
    if len(sents) <= max_n:
        return [s.strip() for s in sents if s.strip()]

    # Score heuristics: position + length + discourse cue words
    scores = []
    n = len(sents)
    for i, s in enumerate(sents):
        pos = 1.0 if (i < 2 or i >= n-2) else 0.5
        ln = min(len(s)/120.0, 1.0)
        cue = 1.2 if re.search(r"\b(however|therefore|importantly|note|tip|steps?)\b", s, re.I) else 1.0
        scores.append((pos*ln*cue, s))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [s.strip() for _, s in scores[:max_n] if s.strip()]


# ------------------------------------------------------------------------------
# Collection Processing
# ------------------------------------------------------------------------------

def process_collection(collection_dir: Path,
                       persona: str,
                       job: str,
                       pdf_filenames: Optional[List[str]]=None,
                       max_sections: int = 10,
                       max_subsects: int = 3) -> Dict[str, Any]:
    """
    Process all PDFs in a single collection directory.
    pdf_filenames: list from input JSON (recommended). If None, scan PDFs/.
    Returns Challenge 1B output structure (dict).
    """
    pdf_dir = collection_dir / "PDFs"
    if pdf_filenames:
        pdf_paths = [pdf_dir / fn for fn in pdf_filenames]
    else:
        pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    existing = [p for p in pdf_paths if p.is_file()]
    if not existing:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    # Pre-build TF-IDF corpus from all doc text to produce consistent feature space.
    corpus_texts = []
    doc_texts_by_stem: Dict[str,List[str]] = {}

    for p in existing:
        pages = extract_pages_pdf(p)
        doc_texts_by_stem[p.stem] = pages
        corpus_texts.extend(pages if pages else [""])

    # Fit vectorizer
    persona_job_txt = preprocess(f"{persona} {job}")
    corpus_for_fit = corpus_texts + [persona_job_txt]
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
    try:
        matrix = tfidf.fit_transform(corpus_for_fit)
        query_vec = matrix[-1:]  # last row
    except Exception as e:
        dprint(f"TF-IDF fit failed: {e}")
        tfidf = None
        query_vec = None

    kw_cache = persona_job_keywords(persona, job)

    # Build sections from each doc
    all_sections: List[Section] = []
    for p in existing:
        pages = doc_texts_by_stem[p.stem]
        # Try 1A outline
        outline_path = locate_1a_outline(p.stem, collection_dir)
        if outline_path:
            outline_obj = load_1a_outline(outline_path)
        else:
            outline_obj = None


        secs = []
        if outline_obj:
            secs = build_sections_from_outline(p.stem, pages, outline_obj)
            if secs:
                dprint(f"{p.name}: using {len(secs)} sections from 1A output.")
            else:
                dprint(f"{p.name}: 1A outline exists but returned no sections, falling back to internal logic.")

        if not secs:
            dprint(f"{p.name}: using internal segmentation logic.")
            for i, txt in enumerate(pages, 1):
                segs = segment_text_fallback(txt)
                for title, body in segs:
                    secs.append(Section(
                        document=p.name,
                        section_title=title,
                        start_page=i,
                        end_page=i,
                        content=body
                    ))
            if not secs:
                # last fallback: whole doc
                secs = [Section(
                    document=p.name,
                    section_title="Full Document",
                    start_page=1,
                    end_page=len(pages),
                    content="\n".join(pages)
                )]
    

        if not secs:
            # fallback: run segmentation over concatenated page text or page by page?
            # We'll segment each page, prefix doc stem
            secs = []
            for i, txt in enumerate(pages, 1):
                segs = segment_text_fallback(txt)
                for title, body in segs:
                    secs.append(Section(
                        document=p.name,
                        section_title=title,
                        start_page=i,
                        end_page=i,
                        content=body
                    ))
            if not secs:
                # last fallback: treat whole doc as single section
                secs = [Section(
                    document=p.name,
                    section_title="Full Document",
                    start_page=1,
                    end_page=len(pages),
                    content="\n".join(pages)
                )]

        # Score
        for s in secs:
            s.relevance = score_section(
                sec_text=s.content,
                persona=persona,
                job=job,
                kw_cache=kw_cache,
                tfidf_vectorizer=tfidf,
                query_vec=query_vec,
            )
        all_sections.extend(secs)

    # Rank across entire collection
    all_sections.sort(key=lambda s: s.relevance, reverse=True)

    # Take top-N
    top = all_sections[:max_sections]

    # Build extracted_sections list
    extracted_sections = []
    for rank, sec in enumerate(top, 1):
        # pick representative page to report: start_page (consistent)
        extracted_sections.append({
            "document": sec.document,
            "section_title": sec.section_title,
            "page_number": sec.start_page,
            "importance_rank": rank
        })

    # Subsection extraction
    subsection_analysis = []
    for rank, sec in enumerate(top, 1):
        subs = pick_top_sentences(sec.content, max_n=max_subsects)
        for j, sent in enumerate(subs, 1):
            subsection_analysis.append({
                "document": sec.document,
                "subsection_id": f"sub_{rank}_{j}",
                "refined_text": sent,
                "page_number": sec.start_page
            })

    # Assemble metadata
    metadata = {
        "input_documents": [p.name for p in existing],
        "persona": persona,
        "job_to_be_done": job,
        "processing_timestamp": datetime.utcnow().isoformat() + "Z",
        # We add challenge_id/test_case_name upstream in driver (we don't know them here)
    }

    return {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }


# ------------------------------------------------------------------------------
# Load collection input JSON
# ------------------------------------------------------------------------------

def load_collection_config(collection_dir: Path) -> Tuple[Dict[str,Any], List[str], str, str]:
    """
    Returns (challenge_info, pdf_filenames, persona_role, job_task).
    Missing fields -> safe defaults; pdf list may be derived from disk if absent.
    """
    cfg_path = collection_dir / "challenge1b_input.json"
    persona_role = "General Analyst"
    job_task = "Analyze the provided documents"
    challenge_info = {}
    pdf_filenames: List[str] = []

    if cfg_path.is_file():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            challenge_info = cfg.get("challenge_info", {})
            persona_role = cfg.get("persona", {}).get("role", persona_role)
            job_task = cfg.get("job_to_be_done", {}).get("task", job_task)
            docs = cfg.get("documents", [])
            pdf_filenames = [d.get("filename") for d in docs if d.get("filename")]
        except Exception as e:
            dprint(f"Config load error ({cfg_path}): {e}")
    else:
        dprint(f"No config at {cfg_path} (using defaults).")

    # if no doc list in config, scan PDFs/ dir
    if not pdf_filenames:
        pdf_filenames = [p.name for p in (collection_dir / "PDFs").glob("*.pdf")]

    return challenge_info, pdf_filenames, persona_role, job_task


# ------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------

def run_for_collection(collection_dir: Path,
                       max_sections: int,
                       max_subsects: int) -> Optional[Path]:
    """
    Run pipeline for a single collection directory.
    Returns output path or None on failure.
    """
    challenge_info, pdf_list, persona_role, job_task = load_collection_config(collection_dir)

    t0 = time.time()
    try:
        result = process_collection(
            collection_dir=collection_dir,
            persona=persona_role,
            job=job_task,
            pdf_filenames=pdf_list,
            max_sections=max_sections,
            max_subsects=max_subsects
        )
        elapsed = time.time() - t0
        result["metadata"]["processing_time_seconds"] = round(elapsed, 2)
        result["metadata"]["challenge_id"] = challenge_info.get("challenge_id", "unknown")
        result["metadata"]["test_case_name"] = challenge_info.get("test_case_name", "unknown")

    except Exception as e:
        dprint(f"Collection processing failed: {e}")
        result = {
            "metadata": {
                "challenge_id": challenge_info.get("challenge_id", "unknown"),
                "test_case_name": challenge_info.get("test_case_name", "unknown"),
                "persona": persona_role,
                "job_to_be_done": job_task,
                "processing_timestamp": datetime.utcnow().isoformat() + "Z",
                "error": str(e),
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }

    out_path = collection_dir / "challenge1b_output.json"
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[Collection] {collection_dir.name} -> {out_path.name} ({len(result['extracted_sections'])} sections).")
    except Exception as e:
        print(f"[Collection] Failed to write {out_path}: {e}")
        return None

    return out_path


# ... (all your original imports and previous code remain the same)

def scan_collections(root: Path) -> List[Path]:
    """
    Scans all subdirectories under root and returns those that:
    - contain 'challenge1b_input.json'
    - and a 'PDFs/' folder with at least one PDF
    """
    candidates = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        input_json = p / "challenge1b_input.json"
        pdf_dir = p / "PDFs"
        if input_json.is_file() and pdf_dir.is_dir() and any(pdf_dir.glob("*.pdf")):
            candidates.append(p)
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Challenge 1B Persona-Driven PDF Analyzer")
    parser.add_argument("--collection", type=str,
                        help="Name of a single collection directory to process (default: process all under root).")
    parser.add_argument("--root", type=str, default=".",
                        help="Root directory containing collection dirs (default: current working dir).")
    parser.add_argument("--max-sections", type=int, default=int(os.getenv("TECHVERSE_MAX_SECTIONS", "10")),
                        help="Max sections in extracted_sections.")
    parser.add_argument("--max-subsects", type=int, default=int(os.getenv("TECHVERSE_MAX_SUBSECTS", "3")),
                        help="Max sentences per section in subsection_analysis.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if args.collection:
        col_dir = (root / args.collection).resolve()
        if not col_dir.is_dir():
            print(f"Collection directory not found: {col_dir}")
            sys.exit(1)
        cols = [col_dir]
    else:
        cols = scan_collections(root)
        if not cols:
            print(f"No valid collection directories found under {root}")
            sys.exit(1)

    print(f"Processing {len(cols)} collection(s)...")
    for c in cols:
        run_for_collection(
            collection_dir=c,
            max_sections=args.max_sections,
            max_subsects=args.max_subsects
        )


if __name__ == "__main__":
    main()
