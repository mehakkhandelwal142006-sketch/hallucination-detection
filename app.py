"""
NLP-Based Hallucination Detection and Reliability Scoring System
=================================================================

Pipeline (document-grounded):

    Trusted source (PDF / Video / Text)
        -> sentence-level evidence units (with page / timestamp)
        -> Sentence-Transformer embeddings + vector search (top-k evidence)
        -> NLI model (entailment / neutral / contradiction)
        -> claim label: Supported / Partially Supported / Contradicted / Unsupported
        -> reliability score + explanation + exportable report

Run:  streamlit run app.py
"""

import base64
import html
import io
import json
import os
import random
import re
import tempfile
import time
from collections import Counter
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from faster_whisper import WhisperModel
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Google GenAI (Gemini) is optional: without it the app still works with pasted answers.
try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types

    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

# python-dotenv is optional: it lets you keep the API key in a local .env file.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# FAISS is optional. If it is not installed we fall back to exact NumPy search.
try:
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


# =========================================================
# CONSTANTS
# =========================================================

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"  # small 6-layer model, CPU friendly

S_SUP = "Supported"
S_PART = "Partially Supported"
S_CON = "Contradicted"
S_UNS = "Unsupported"
ALL_STATUSES = [S_SUP, S_PART, S_CON, S_UNS]

STATUS_META = {
    S_SUP: {"color": "#22c55e", "icon": "✅"},
    S_PART: {"color": "#eab308", "icon": "⚠️"},
    S_CON: {"color": "#ef4444", "icon": "⛔"},
    S_UNS: {"color": "#f97316", "icon": "❓"},
}

# Reliability weights (documented in the UI as well)
WEIGHTS = {S_SUP: 1.0, S_PART: 0.5, S_CON: 0.0, S_UNS: 0.0}

METHOD_HYBRID = "Hybrid (Similarity + NLI)"
METHOD_NLI = "NLI only"
METHOD_SIM = "Similarity only"
METHODS = [METHOD_HYBRID, METHOD_NLI, METHOD_SIM]

STOPWORDS = {
    "that", "this", "with", "from", "have", "been", "were", "will", "would",
    "could", "should", "about", "which", "their", "there", "they", "them",
    "then", "than", "also", "into", "over", "such", "some", "more", "most",
    "very", "what", "when", "where", "while", "does", "used", "using", "uses",
    "each", "other", "these", "those", "being", "because", "however",
    "between", "through", "during", "before", "after", "under", "again",
    "just", "only", "same", "both", "many", "much", "like", "make", "made",
    "is", "are", "was",
}


# =========================================================
# MODEL LOADERS (cached so they load only once)
# =========================================================

@st.cache_resource(show_spinner=False)
def load_whisper_model(size: str):
    return WhisperModel(size, device="cpu", compute_type="int8")


@st.cache_resource(show_spinner=False)
def load_semantic_model():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def load_nli_model():
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    model.eval()
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    return tokenizer, model, labels


# =========================================================
# TEXT PROCESSING
# =========================================================

def wrap_long(text, limit=350):
    """Break very long 'sentences' (e.g. bullet lists in slides) into chunks."""
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for word in text.split():
        if len(current) + len(word) + 1 > limit and current:
            chunks.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        chunks.append(current)
    return chunks


def split_sentences(text, min_len=25):
    """Split raw text into clean sentences (handles PDF line-wrapping)."""
    text = re.sub(r"-\s*\n\s*(?=[a-z])", "", text)  # re-join hyphenated words
    text = re.sub(r"\s*\n\s*", " ", text)            # PDF hard line breaks
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“‘(\[])', text)

    sentences = []
    for part in parts:
        for chunk in re.split(r"\s[•●▪◦·]\s|;\s", part):
            sentences.extend(wrap_long(chunk.strip()))

    return [s for s in sentences if len(s) >= min_len]


def extract_claims(raw_text):
    """Return (claims, skipped_questions) from the user's input."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", raw_text.strip())

    claims, skipped, seen = [], [], set()
    for part in parts:
        part = re.sub(r"^\s*(\d+[.)]|[-•*])\s+", "", part).strip()
        if len(part) <= 5:
            continue
        if part.endswith("?"):
            skipped.append(part)
            continue
        if part.lower() not in seen:
            seen.add(part.lower())
            claims.append(part)
    return claims, skipped


def fmt_time(seconds):
    seconds = int(seconds or 0)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def short_name(name, limit=26):
    """Shorten long file names for display: 'Artificial_intelligence_exp….webm'."""
    root, ext = os.path.splitext(name)
    return name if len(root) <= limit else root[: limit - 1] + "…" + ext


def cite(unit):
    name = short_name(unit["source"])
    return f"{name} · {unit['loc']}" if unit.get("loc") else name


# =========================================================
# SOURCE LOADING (PDF / VIDEO)
# =========================================================

@st.cache_data(show_spinner=False)
def pdf_units(name, data):
    """Extract sentence units (with page numbers) from a PDF."""
    reader = PdfReader(io.BytesIO(data))
    units = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for sentence in split_sentences(text):
            units.append({"text": sentence, "source": name, "loc": f"p. {page_no}"})
    return units, len(reader.pages)


def segments_to_units(segments, source):
    """Merge Whisper segments into sentences, keeping the start timestamp."""
    units, buffer, start = [], [], None

    def flush():
        nonlocal buffer, start
        text = " ".join(buffer).strip()
        if text:
            for sentence in split_sentences(text, min_len=10):
                units.append(
                    {"text": sentence, "source": source, "loc": fmt_time(start)}
                )
        buffer, start = [], None

    for seg in segments:
        piece = seg.text.strip()
        if not piece:
            continue
        if start is None:
            start = seg.start
        buffer.append(piece)
        if re.search(r'[.!?]["\')\]]*$', piece):
            flush()
    flush()
    return units


def transcribe_video(video_file, model_size):
    """Transcribe an uploaded video. Returns (sentence units, language)."""
    temp_path = None
    try:
        suffix = os.path.splitext(video_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_file.getvalue())
            temp_path = tmp.name

        model = load_whisper_model(model_size)
        segments, info = model.transcribe(temp_path, beam_size=5, vad_filter=True)
        units = segments_to_units(segments, video_file.name)
        return units, info.language
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# =========================================================
# RETRIEVAL (embeddings + vector search)
# =========================================================

@st.cache_data(show_spinner=False)
def embed_texts(texts: tuple):
    model = load_semantic_model()
    return model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )


def build_index(embeddings):
    if HAS_FAISS:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype("float32"))
        return index
    return embeddings


def search_index(index, query_emb, k):
    if HAS_FAISS:
        k = min(k, index.ntotal)
        scores, ids = index.search(query_emb.astype("float32"), k)
        return ids[0], scores[0]
    k = min(k, len(index))
    sims = index @ query_emb[0]
    ids = np.argsort(-sims)[:k]
    return ids, sims[ids]


def answer_question(question, units, index, k=6, min_sim=0.30):
    """Answer a question using the most relevant sentences of the trusted source."""
    query = embed_texts((question,))
    ids, sims = search_index(index, query, k)
    best = float(sims[0])

    if best < min_sim:
        return {"found": False, "text": "", "hits": []}

    picked = {}  # position in `units` -> relevance
    for i, sim in zip(ids, sims):
        i, sim = int(i), float(sim)
        if sim < min_sim or sim < best - 0.20:
            continue
        if units[i]["text"].rstrip().endswith("?"):
            # A question / heading inside the source is not an answer:
            # the answer is the sentence(s) that follow it.
            for j in range(i + 1, min(i + 3, len(units))):
                if units[j]["source"] != units[i]["source"] or units[j]["text"].rstrip().endswith("?"):
                    break
                picked.setdefault(j, sim)
        else:
            picked.setdefault(i, sim)

    # Keep the best 4 sentences, but show them in the order they appear in the source
    top = sorted(picked, key=lambda pos: -picked[pos])[:4]
    hits = [
        {"citation": cite(units[pos]), "sim": picked[pos], "text": units[pos]["text"]}
        for pos in sorted(top)
    ]

    return {"found": True, "text": " ".join(h["text"] for h in hits), "hits": hits}


# =========================================================
# GEMINI (Google GenAI) — generate an answer that will be verified
# =========================================================

GEN_LIMIT = 8  # max Gemini calls per browser session (protects the free quota)

# Used only if the live model list cannot be read. Override with GEMINI_MODEL in secrets / .env.
FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]

GEN_INSTRUCTION = (
    "Answer the user's question from your general knowledge in 4 to 7 short, factual sentences. "
    "Write each sentence on its own line. "
    "Do not use bullet points, numbering, markdown, headings or quotation marks. "
    "Do not use pronouns such as 'it', 'this' or 'they' for the topic: repeat the subject by name "
    "so that every sentence makes sense on its own. "
    "Do not add an introduction or a conclusion."
)


def get_secret(name):
    """Read a setting from Streamlit secrets (cloud) or environment / .env (local)."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return (os.environ.get(name) or "").strip()


@st.cache_resource(ttl=3600, show_spinner=False)
def gemini_client(api_key):
    return genai.Client(api_key=api_key)


def choose_models(client):
    """Pick Gemini models to try: GEMINI_MODEL override, else newest Flash from Google's live list."""
    override = get_secret("GEMINI_MODEL")
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]

    found = []
    try:
        for m in client.models.list():
            name = (m.name or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if actions and "generateContent" not in actions:
                continue
            match = re.fullmatch(r"gemini-(\d+(?:\.\d+)*)-flash(-lite)?", name)
            if match:
                version = tuple(int(x) for x in match.group(1).split("."))
                found.append((version, match.group(2) is None, name))
    except Exception:
        pass

    found.sort(reverse=True)  # newest version first, plain Flash before Flash-Lite
    names = [n for _, _, n in found]
    return names + [m for m in FALLBACK_MODELS if m not in names]


def generate_answer(question, api_key):
    """Ask Gemini to answer the question. Returns (text, model_name)."""
    client = gemini_client(api_key)
    config = genai_types.GenerateContentConfig(system_instruction=GEN_INSTRUCTION)

    models = choose_models(client)
    last = st.session_state.get("gemini_model")
    if last in models:
        models.remove(last)
        models.insert(0, last)

    problem, last_err, tried = "", None, []
    for model in models[:4]:
        response = None
        for attempt in range(2):  # one quick retry when Google is temporarily overloaded
            try:
                response = client.models.generate_content(
                    model=model, contents=question, config=config
                )
                break
            except genai_errors.APIError as err:
                code = getattr(err, "code", None)
                if code in (500, 502, 503, 504) and attempt == 0:
                    time.sleep(2)
                    continue
                if code in (404, 429, 500, 502, 503, 504):
                    # model missing / overloaded / out of quota -> try the next model
                    last_err = err
                    tried.append(model)
                    break
                raise  # e.g. invalid API key: no point trying other models

        if response is None:
            continue

        text = (response.text or "").strip()
        if text:
            st.session_state["gemini_model"] = model
            return text, model
        problem = "Gemini returned an empty answer (it may have been blocked by a safety filter)."
        break

    if last_err is not None:
        last_err.tried = tried
        raise last_err
    raise RuntimeError(problem or "No Gemini model was available.")


def clean_generated(text):
    """Turn Gemini's reply into plain lines: no markdown, bullets, numbering or questions."""
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"[*`]+", "", raw)
        line = re.sub(r"^\s*(#+|>|[-•]|\d+[.)])\s*", "", line).strip()
        if len(line) > 5 and not line.endswith("?"):
            lines.append(line)
    return "\n".join(lines[:10])


def friendly_error(err):
    code = getattr(err, "code", None)
    text = str(err)
    tried = getattr(err, "tried", None)
    suffix = f" (models tried: {', '.join(tried)})" if tried else ""
    if code == 429:
        return "Gemini's free quota is used up for now (too many requests). Wait a minute and try again, or paste an answer yourself." + suffix
    if code in (401, 403) or "API key" in text or "API_KEY" in text:
        return "Gemini did not accept the API key. Check GEMINI_API_KEY in Secrets (cloud) or in the .env file (local)."
    if code in (500, 502, 503, 504):
        return f"Gemini is overloaded right now (error {code}). Please try again in a minute." + suffix
    if isinstance(err, RuntimeError):
        return text
    return f"Gemini request failed{f' ({code})' if code else ''}: {text[:200]}"


# =========================================================
# NLI VERIFICATION
# =========================================================

def nli_scores(premises, hypothesis):
    """Return [{'entailment':p, 'neutral':p, 'contradiction':p}, ...]."""
    tokenizer, model, labels = load_nli_model()
    inputs = tokenizer(
        premises,
        [hypothesis] * len(premises),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1).numpy()
    return [{labels[i]: float(row[i]) for i in range(len(row))} for row in probs]


def label_similarity(sim, cfg):
    if sim >= cfg["sim_sup"]:
        return S_SUP
    if sim >= cfg["sim_part"]:
        return S_PART
    return S_UNS


def label_nli(ent, con, cfg):
    if ent >= cfg["t_sup"] and ent >= con:
        return S_SUP
    if con >= cfg["t_con"]:
        return S_CON
    if ent >= cfg["t_part"]:
        return S_PART
    return S_UNS


def build_vocab(texts):
    words = set()
    for t in texts:
        words.update(re.findall(r"[a-z0-9]+", t.lower()))
    return {"words": words, "prefixes": {w[:5] for w in words if not w.isdigit()}}


def find_missing_terms(claim, vocab):
    """Key words / numbers of the claim that never appear in the trusted source."""
    missing = []
    for term in re.findall(r"[a-z0-9]+", claim.lower()):
        if term.isdigit():
            if len(term) >= 2 and term not in vocab["words"]:
                missing.append(term)
        elif len(term) >= 4 and term not in STOPWORDS:
            if term[:5] not in vocab["prefixes"]:
                missing.append(term)
    return list(dict.fromkeys(missing))


def build_explanation(status, ev, missing):
    where = ev["citation"]
    if status == S_SUP:
        text = (
            f"Supported: the source ({where}) says the same thing "
            f"(entailment {ev['entail']:.0%})."
        )
    elif status == S_PART:
        text = (
            f"Partially supported: the source ({where}) backs part of the claim "
            f"(entailment {ev['entail']:.0%}), but some details are not confirmed."
        )
    elif status == S_CON:
        text = (
            f"Contradicted: the source ({where}) says the opposite "
            f"(contradiction {ev['contra']:.0%})."
        )
    elif ev["neutral"] >= max(ev["entail"], ev["contra"]):
        text = (
            f"Neutral, so not supported: the closest sentence ({where}) is about the "
            f"same topic but does not confirm or deny the claim (neutral {ev['neutral']:.0%})."
        )
    else:
        text = (
            f"Not supported: no sentence in the source confirms this claim. "
            f"Closest match: {where} (similarity {ev['sim']:.2f})."
        )
    if missing and status != S_SUP:
        text += f" Not found anywhere in the source: {', '.join(missing)}."
    return text


def verify_claim(claim_id, claim, question, units, index, vocab, cfg):
    # The question (if given) gives extra context, e.g. for claims like "It was invented in 1956."
    query_text = f"{question} {claim}".strip() if question else claim
    query = embed_texts((query_text,))
    ids, sims = search_index(index, query, cfg["top_k"])

    premises = [units[int(i)]["text"] for i in ids]
    probs = nli_scores(premises, claim)

    cands = []
    for i, sim, p in zip(ids, sims, probs):
        unit = units[int(i)]
        cands.append(
            {
                "text": unit["text"],
                "citation": cite(unit),
                "sim": float(sim),
                "entail": p.get("entailment", 0.0),
                "neutral": p.get("neutral", 0.0),
                "contra": p.get("contradiction", 0.0),
            }
        )

    top = cands[0]
    best_ent = max(cands, key=lambda c: c["entail"])
    best_con = max(cands, key=lambda c: c["contra"])

    s_sim = label_similarity(top["sim"], cfg)
    s_nli = label_nli(best_ent["entail"], best_con["contra"], cfg)

    # Hybrid = NLI verdict, but only trusted when retrieval found relevant text,
    # plus a lexical sanity check (numbers that never appear in the source).
    missing = find_missing_terms(claim, vocab)
    s_hyb = s_nli
    if s_nli in (S_SUP, S_PART) and best_ent["sim"] < cfg["sim_floor"]:
        s_hyb = S_UNS
    elif s_nli == S_CON and best_con["sim"] < cfg["sim_floor"]:
        s_hyb = S_UNS
    if s_hyb == S_SUP and any(t.isdigit() for t in missing):
        s_hyb = S_PART

    by_method = {METHOD_HYBRID: s_hyb, METHOD_NLI: s_nli, METHOD_SIM: s_sim}
    status = by_method[cfg["method"]]

    if cfg["method"] == METHOD_SIM:
        ev = top
    elif status in (S_SUP, S_PART):
        ev = best_ent
    elif status == S_CON:
        ev = best_con
    else:
        ev = top

    return {
        "id": claim_id,
        "claim": claim,
        "status": status,
        "status_sim": s_sim,
        "status_nli": s_nli,
        "status_hybrid": s_hyb,
        "sim": ev["sim"],
        "entail": ev["entail"],
        "neutral": ev["neutral"],
        "contra": ev["contra"],
        "evidence": ev["text"],
        "citation": ev["citation"],
        "missing": missing,
        "explanation": build_explanation(status, ev, missing),
        "candidates": cands,
    }


def reliability_score(statuses):
    statuses = list(statuses)
    if not statuses:
        return 0.0
    return sum(WEIGHTS[s] for s in statuses) / len(statuses)


# =========================================================
# UI HELPERS
# =========================================================

esc = html.escape


def kpi(number, label, color=None):
    style = f' style="color:{color}"' if color else ""
    return (
        '<div class="kpi">'
        f'<div class="kpi-num"{style}>{number}</div>'
        f'<div class="kpi-label">{label}</div>'
        "</div>"
    )


def prob_bar(label, value, color):
    return (
        '<div class="pbar">'
        f'<span class="pbar-label">{label}</span>'
        '<div class="pbar-track">'
        f'<div class="pbar-fill" style="width:{value * 100:.0f}%;background:{color}"></div>'
        "</div>"
        f'<span class="pbar-val">{value:.0%}</span>'
        "</div>"
    )


def claim_card(row):
    meta = STATUS_META[row["status"]]
    color = meta["color"]
    chips = ""
    if row["missing"] and row["status"] != S_SUP:
        chips = (
            '<div class="chip-row"><span class="chip-title">Not found in source:</span>'
            + "".join(f'<span class="chip">{esc(t)}</span>' for t in row["missing"][:8])
            + "</div>"
        )
    return (
        f'<div class="claim-card" style="border-left:6px solid {color}">'
        '<div class="claim-head">'
        f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}66">'
        f'{meta["icon"]} {row["status"]}</span>'
        f'<span class="claim-id">Claim {row["id"]}</span>'
        "</div>"
        f'<div class="claim-text">{esc(row["claim"])}</div>'
        f'<div class="claim-expl">{esc(row["explanation"])}</div>'
        f"{chips}"
        '<div class="pbars">'
        + prob_bar("Entailment", row["entail"], "#22c55e")
        + prob_bar("Neutral", row["neutral"], "#94a3b8")
        + prob_bar("Contradiction", row["contra"], "#ef4444")
        + "</div>"
        "</div>"
    )


def summary_row(row):
    meta = STATUS_META[row["status"]]
    color = meta["color"]
    return (
        f'<div class="sum-row" style="border-left:5px solid {color}">'
        f'<div class="sum-num">{row["id"]}</div>'
        '<div class="sum-body">'
        f'<div class="sum-claim">{esc(row["claim"])}</div>'
        f'<div class="sum-expl">{esc(row["explanation"])}</div>'
        "</div>"
        f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}66">'
        f'{meta["icon"]} {row["status"]}</span>'
        "</div>"
    )


def answer_card(ans):
    head = (
        f'<div class="ans-title">💡 ANSWER FROM THE SOURCE</div>'
        f'<div class="ans-q">❓ {esc(ans["question"])}</div>'
    )
    if not ans["found"]:
        return (
            '<div class="ans-card" style="border-left:6px solid #f97316">'
            + head
            + '<div class="ans-text">No clear answer to this question was found in the trusted source.</div>'
            "</div>"
        )
    places = list(dict.fromkeys(h["citation"] for h in ans["hits"]))
    chips = "".join(f'<span class="chip src">{esc(c)}</span>' for c in places)
    return (
        '<div class="ans-card" style="border-left:6px solid #22c55e">'
        + head
        + f'<div class="ans-text">{esc(ans["text"])}</div>'
        + f'<div class="chip-row"><span class="chip-title">Taken from:</span>{chips}</div>'
        "</div>"
    )


def ring(pct, color):
    return (
        f'<div class="ring" style="--p:{pct};--c:{color}">'
        '<div class="ring-text">'
        f'<div class="ring-num">{pct:.1f}%</div>'
        '<div class="ring-lbl">Reliability</div>'
        "</div></div>"
    )


def glass(key):
    """A bordered container; the key lets the CSS style it as a glass card."""
    try:
        return st.container(border=True, key=key)
    except TypeError:  # older Streamlit without container keys
        return st.container(border=True)


EXTRA_STOP = {
    "your", "every", "often", "well", "even", "within", "without", "across",
    "various", "include", "includes", "including", "called", "known", "since",
    "therefore", "thus", "whose", "first", "second", "third", "based", "given",
    "help", "helps", "need", "needs", "come", "comes", "work", "works", "take",
    "takes", "more", "less", "type", "types",
}


def top_topics(units, n=12):
    """Most frequent meaningful words in the trusted source."""
    counts = Counter()
    for u in units:
        for w in re.findall(r"[a-zA-Z]{4,}", u["text"].lower()):
            if w not in STOPWORDS and w not in EXTRA_STOP:
                counts[w] += 1
    return [w for w, _ in counts.most_common(n)]


def source_panel(units):
    words = sum(len(u["text"].split()) for u in units)
    files = len({u["source"] for u in units})
    chips = "".join(f'<span class="topic">{esc(t)}</span>' for t in top_topics(units))
    return (
        '<div class="stat-strip">'
        f'<div class="stat"><b>{len(units)}</b><span>evidence sentences</span></div>'
        f'<div class="stat"><b>{words:,}</b><span>words</span></div>'
        f'<div class="stat"><b>{files}</b><span>source file(s)</span></div>'
        "</div>"
        f'<div class="chip-row"><span class="chip-title">Key topics in your source:</span>{chips}</div>'
    )


def highlight_evidence(evidence, claim):
    """Wrap the words of the evidence that also appear in the claim in <mark>."""
    terms = {
        t[:5]
        for t in re.findall(r"[a-z0-9]+", claim.lower())
        if (t.isdigit() or len(t) >= 4) and t not in STOPWORDS
    }
    out = []
    for token in re.split(r"(\W+)", evidence):
        if token and re.fullmatch(r"[A-Za-z0-9]+", token) and token.lower()[:5] in terms \
                and token.lower() not in STOPWORDS:
            out.append(f'<mark class="hl">{esc(token)}</mark>')
        else:
            out.append(esc(token))
    return "".join(out)


REPORT_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #1f2937; }
h1 { margin-bottom: 4px; }
.muted { color: #6b7280; font-size: 13px; }
.kpis { display: flex; gap: 10px; margin: 18px 0; flex-wrap: wrap; }
.kpi { flex: 1; min-width: 110px; border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; text-align: center; font-size: 12px; }
.kpi b { display: block; font-size: 24px; }
table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }
th, td { border: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }
th { background: #f3f4f6; }
.box { border: 1px solid #e5e7eb; border-left: 5px solid #6366f1; border-radius: 8px; padding: 10px 14px; margin: 10px 0; }
.q { font-weight: 700; margin-bottom: 4px; }
"""


def build_html_report(analysis, counts, score_pct, halluc_rate):
    """A clean, printable report (open in a browser -> Print -> Save as PDF)."""

    def badge(status):
        c = STATUS_META[status]["color"]
        return (
            f'<span style="background:{c}22;color:{c};border:1px solid {c};'
            f'padding:2px 10px;border-radius:999px;font-weight:700;font-size:12px">{esc(status)}</span>'
        )

    answers_html = ""
    for a in analysis.get("answers", []):
        body = esc(a["text"]) if a["found"] else "No clear answer was found in the trusted source."
        answers_html += (
            f'<div class="box"><div class="q">Q: {esc(a["question"])}</div><div>{body}</div></div>'
        )

    rows_html = "".join(
        "<tr>"
        f'<td>{r["id"]}</td><td>{esc(r["claim"])}</td><td>{badge(r["status"])}</td>'
        f'<td>{esc(r["explanation"])}</td><td>{esc(r["citation"])}</td>'
        "</tr>"
        for r in analysis["rows"]
    )

    kpis = "".join(
        f'<div class="kpi"><b>{value}</b>{label}</div>'
        for label, value in [
            ("Reliability", f"{score_pct}%"),
            ("Hallucination rate", f"{halluc_rate}%"),
            ("Total claims", len(analysis["rows"])),
            ("Supported", counts[S_SUP]),
            ("Partially", counts[S_PART]),
            ("Contradicted", counts[S_CON]),
            ("Unsupported", counts[S_UNS]),
        ]
    )

    answers_section = f"<h2>Answers from the source</h2>{answers_html}" if answers_html else ""

    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Hallucination Report</title>"
        f"<style>{REPORT_CSS}</style></head><body>"
        "<h1>Hallucination Detection Report</h1>"
        f'<div class="muted">Source: {esc(analysis["source"])} · {esc(analysis["time"])}</div>'
        f'<div class="kpis">{kpis}</div>'
        f"{answers_section}"
        "<h2>Claim verification</h2>"
        "<table><tr><th>#</th><th>Claim</th><th>Status</th><th>Explanation</th><th>Evidence location</th></tr>"
        f"{rows_html}</table>"
        '<p class="muted">Tip: open this file in a browser and use Print → Save as PDF.</p>'
        "</body></html>"
    )


# ---------------------------------------------------------------
# Background: your own image (background.jpg / .png / .webp next to app.py)
# or an automatically generated "neural network" picture.
# ---------------------------------------------------------------

def network_svg_b64():
    rnd = random.Random(7)
    width, height = 1600, 900
    colors = ["#818cf8", "#c084fc", "#f472b6", "#22d3ee"]
    pts = [
        (rnd.randint(0, width), rnd.randint(0, height), rnd.choice(colors), rnd.choice([2, 2, 3, 3, 4, 6]))
        for _ in range(75)
    ]

    lines = []
    for i, (x1, y1, _, _) in enumerate(pts):
        for x2, y2, _, _ in pts[i + 1:]:
            d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if d < 210:
                lines.append(
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#a78bfa" '
                    f'stroke-opacity="{0.30 * (1 - d / 210):.3f}" stroke-width="1"/>'
                )

    dots = "".join(
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}" fill-opacity=".85"/>' for x, y, c, r in pts
    )
    glows = "".join(
        f'<circle cx="{x}" cy="{y}" r="{r * 5}" fill="{c}" fill-opacity=".12" filter="url(#b)"/>'
        for x, y, c, r in pts
        if r >= 4
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'preserveAspectRatio="xMidYMid slice"><defs><filter id="b">'
        '<feGaussianBlur stdDeviation="8"/></filter></defs>'
        f'{glows}{"".join(lines)}{dots}</svg>'
    )
    return base64.b64encode(svg.encode("utf-8")).decode("ascii")


BG_TEMPLATE = """
<style>
.stApp {
    background-color: #070916;
    background-image:
        radial-gradient(circle at 12% 8%, rgba(124,58,237,.32), transparent 38%),
        radial-gradient(circle at 88% 16%, rgba(219,39,119,.24), transparent 40%),
        radial-gradient(circle at 50% 105%, rgba(34,211,238,.14), transparent 45%),
        __OVERLAY__,
        __IMAGE__;
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
</style>
"""


@st.cache_data(show_spinner=False)
def background_css():
    try:
        folders = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    except NameError:
        folders = [os.getcwd()]

    mimes = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    names = [f"{base}.{ext}" for base in ("background", "bg") for ext in mimes]

    for folder in folders:
        for name in names:
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.getsize(path) < 4_000_000:
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("ascii")
                mime = mimes[name.rsplit(".", 1)[1]]
                return (
                    BG_TEMPLATE
                    .replace("__OVERLAY__", "linear-gradient(rgba(6,8,20,.72), rgba(6,8,20,.86))")
                    .replace("__IMAGE__", f"url(data:{mime};base64,{data})")
                )

    return (
        BG_TEMPLATE
        .replace("__OVERLAY__", "linear-gradient(rgba(7,9,22,.30), rgba(7,9,22,.50))")
        .replace("__IMAGE__", f"url(data:image/svg+xml;base64,{network_svg_b64()})")
    )


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&display=swap');

.block-container { padding-top: 3rem; max-width: 1250px; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
[data-testid="stDeployButton"], [data-testid="stAppDeployButton"], .stDeployButton { display: none !important; }

/* charts: rounded glass look instead of a plain black box */
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {
    border-radius: 18px; overflow: hidden; padding: 6px;
    background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.10);
}

h2, h3, h4, .hero-title, .step-title, .kpi-num, .ring-num, .ans-q, .claim-text, .sum-claim {
    font-family: 'Poppins', 'Segoe UI', sans-serif;
}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: rgba(9,11,30,.72);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border-right: 1px solid rgba(255,255,255,.08);
}
section[data-testid="stSidebar"] > div:first-child { background: transparent; }

/* ---------- animations ---------- */
@keyframes heroShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes floaty { 0%, 100% { transform: translate(0, 0); } 50% { transform: translate(-14px, 18px); } }
@keyframes bob { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
@keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
@property --p { syntax: '<number>'; inherits: true; initial-value: 0; }
@keyframes ringfill { from { --p: 0; } }

/* ---------- hero ---------- */
.hero {
    position: relative; overflow: hidden; padding: 36px 40px; border-radius: 26px; color: #fff; margin-bottom: 24px;
    background: linear-gradient(120deg, #312e81, #6d28d9, #be185d, #4f46e5);
    background-size: 300% 300%; animation: heroShift 14s ease infinite;
    border: 1px solid rgba(255,255,255,.2); box-shadow: 0 24px 70px rgba(124,58,237,.38);
}
.orb { position: absolute; border-radius: 50%; filter: blur(32px); opacity: .55; animation: floaty 9s ease-in-out infinite; }
.o1 { width: 230px; height: 230px; background: #22d3ee; right: -60px; top: -90px; }
.o2 { width: 170px; height: 170px; background: #f472b6; right: 220px; bottom: -100px; animation-delay: -3s; }
.o3 { width: 130px; height: 130px; background: #a78bfa; left: 42%; top: -70px; animation-delay: -6s; }
.hero-inner { position: relative; z-index: 1; }
.hero-tag {
    display: inline-block; font-size: 12px; letter-spacing: .1em; font-weight: 700; margin-bottom: 14px;
    background: rgba(255,255,255,.18); padding: 5px 14px; border-radius: 999px; border: 1px solid rgba(255,255,255,.25);
}
.hero-title { font-size: 42px; font-weight: 800; line-height: 1.15; margin-bottom: 8px; text-shadow: 0 2px 24px rgba(0,0,0,.35); }
.hero-title .grad {
    background: linear-gradient(90deg, #67e8f9, #f9a8d4); -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.shield { display: inline-block; animation: bob 3s ease-in-out infinite; }
.hero-sub { font-size: 16px; opacity: .93; max-width: 760px; }
.pipeline { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 20px; }
.pipe-step {
    background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
    padding: 7px 16px; border-radius: 999px; font-size: 13px; font-weight: 600; backdrop-filter: blur(6px);
}
.pipe-step b {
    display: inline-flex; width: 20px; height: 20px; border-radius: 50%; align-items: center; justify-content: center;
    background: #fff; color: #6d28d9; font-size: 12px; margin-right: 7px;
}
.pipe-arrow { opacity: .8; font-weight: 700; }

/* ---------- glass cards ---------- */
.st-key-card_source, .st-key-card_ask {
    background: rgba(255,255,255,.05) !important;
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,.13) !important; border-radius: 22px !important;
    box-shadow: 0 14px 44px rgba(0,0,0,.35); padding: 8px 10px; transition: border-color .2s, box-shadow .2s;
}
.st-key-card_source:hover, .st-key-card_ask:hover {
    border-color: rgba(167,139,250,.55) !important; box-shadow: 0 14px 50px rgba(124,58,237,.30);
}

.step-title { font-size: 22px; font-weight: 700; margin: 4px 0 12px 0; display: flex; align-items: center; gap: 12px; }
.step-num {
    width: 34px; height: 34px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;
    font-size: 16px; font-weight: 800; color: #fff; background: linear-gradient(135deg, #6366f1, #ec4899);
    box-shadow: 0 0 18px rgba(236,72,153,.55);
}

/* ---------- inputs & buttons ---------- */
div[data-baseweb="textarea"], div[data-baseweb="input"] {
    background: rgba(255,255,255,.05) !important; border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,.14) !important;
}
div[data-baseweb="textarea"] textarea, div[data-baseweb="input"] input { background: transparent !important; }
section[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,.04); border: 1.5px dashed rgba(167,139,250,.55); border-radius: 16px;
}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899) !important; border: 0 !important; color: #fff !important;
    font-weight: 700; letter-spacing: .02em; padding: .75rem 1rem; border-radius: 16px;
    box-shadow: 0 10px 34px rgba(168,85,247,.45); transition: transform .15s, box-shadow .15s;
}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px); box-shadow: 0 14px 44px rgba(236,72,153,.55);
}
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
    border-radius: 12px; border: 1px solid rgba(255,255,255,.2); background: rgba(255,255,255,.06);
}
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover { border-color: #a855f7; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs button[data-baseweb="tab"] {
    background: rgba(255,255,255,.05); border-radius: 12px 12px 0 0; padding: 8px 18px; font-weight: 600;
}
.stTabs button[data-baseweb="tab"][aria-selected="true"] { background: rgba(139,92,246,.25); }
.stTabs [data-baseweb="tab-highlight"] { background: #a855f7; height: 3px; }

/* ---------- source panel ---------- */
.stat-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 12px 0; }
.stat {
    text-align: center; padding: 12px 8px; border-radius: 14px;
    background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.12);
}
.stat b {
    display: block; font-size: 24px; font-weight: 800; font-family: 'Poppins', sans-serif;
    background: linear-gradient(90deg, #67e8f9, #f9a8d4); -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat span { font-size: 12px; opacity: .75; }
.topic {
    font-size: 12px; padding: 3px 12px; border-radius: 999px; font-weight: 600;
    background: linear-gradient(90deg, rgba(99,102,241,.28), rgba(236,72,153,.28)); border: 1px solid rgba(255,255,255,.18);
}

/* ---------- result cards ---------- */
.kpi, .claim-card, .sum-row, .ans-card {
    background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.12);
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    animation: fadeUp .5s ease both; transition: transform .18s, box-shadow .18s, border-color .18s;
}
.kpi:hover, .claim-card:hover, .sum-row:hover, .ans-card:hover {
    transform: translateY(-3px); box-shadow: 0 14px 36px rgba(0,0,0,.4); border-color: rgba(167,139,250,.5);
}

.kpi { border-radius: 16px; padding: 16px 10px; text-align: center; border-top: 3px solid rgba(167,139,250,.7); }
.kpi-num { font-size: 32px; font-weight: 800; line-height: 1.1; }
.kpi-label { font-size: 13px; opacity: .78; margin-top: 4px; }

.ring { position: relative; width: 190px; height: 190px; margin: 6px auto; filter: drop-shadow(0 0 16px var(--c)); }
.ring::before {
    content: ""; position: absolute; inset: 0; border-radius: 50%; animation: ringfill 1.4s ease-out;
    background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255,255,255,.14) 0);
    -webkit-mask: radial-gradient(farthest-side, transparent 70%, #000 71%);
    mask: radial-gradient(farthest-side, transparent 70%, #000 71%);
}
.ring-text { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.ring-num { font-size: 34px; font-weight: 800; }
.ring-lbl { font-size: 13px; opacity: .72; }

.claim-card { border-radius: 16px; padding: 16px 18px; margin: 12px 0 4px 0; }
.claim-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.badge { font-size: 13px; font-weight: 700; padding: 3px 12px; border-radius: 999px; }
.claim-id { font-size: 13px; opacity: .6; font-weight: 600; }
.claim-text { font-size: 17px; font-weight: 600; margin-bottom: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.claim-expl { font-size: 14px; opacity: .85; margin-bottom: 10px; }

.chip-row { margin-bottom: 10px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.chip-title { font-size: 12px; opacity: .72; margin-right: 4px; }
.chip {
    font-size: 12px; padding: 2px 10px; border-radius: 999px; font-weight: 600;
    background: rgba(239,68,68,.15); color: #f87171; border: 1px solid rgba(239,68,68,.4);
}
.chip.src { background: rgba(99,102,241,.18); color: #a5b4fc; border-color: rgba(99,102,241,.45); }

.pbars { display: grid; gap: 5px; }
.pbar { display: flex; align-items: center; gap: 10px; font-size: 12px; }
.pbar-label { width: 90px; opacity: .78; }
.pbar-track { flex: 1; height: 8px; border-radius: 999px; background: rgba(255,255,255,.14); overflow: hidden; }
.pbar-fill { height: 100%; border-radius: 999px; }
.pbar-val { width: 40px; text-align: right; font-weight: 600; }

.sum-row { display: flex; align-items: flex-start; gap: 12px; margin: 8px 0; padding: 12px 14px; border-radius: 14px; }
.sum-num {
    min-width: 30px; height: 30px; border-radius: 50%; background: linear-gradient(135deg, #6366f1, #ec4899);
    display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 13px; color: #fff;
}
.sum-body { flex: 1; min-width: 0; }
.sum-claim { font-weight: 600; font-size: 15px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sum-expl { font-size: 13px; opacity: .82; margin-top: 3px; }
.sum-row .badge { white-space: nowrap; flex-shrink: 0; }

.ans-card { border-radius: 18px; padding: 18px 20px; margin: 12px 0 4px 0; }
.ans-title { font-size: 12px; letter-spacing: .08em; font-weight: 700; opacity: .7; margin-bottom: 6px; }
.ans-q { font-weight: 700; font-size: 17px; margin-bottom: 8px; }
.ans-text { font-size: 15px; line-height: 1.6; margin-bottom: 10px; }

.ev-box {
    border-left: 4px solid #8b5cf6; background: rgba(139,92,246,.12); padding: 12px 14px;
    border-radius: 10px; line-height: 1.65; margin: 6px 0;
}
mark.hl { background: rgba(250,204,21,.25); color: #fde68a; padding: 0 3px; border-radius: 4px; }

.footer { text-align: center; opacity: .6; font-size: 13px; margin-top: 28px; padding-bottom: 12px; }
</style>
"""


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Hallucination Detection System",
    page_icon="🛡️",
    layout="wide",
)
st.markdown(CSS + background_css(), unsafe_allow_html=True)

st.markdown(
    """
<div class="hero">
<div class="orb o1"></div><div class="orb o2"></div><div class="orb o3"></div>
<div class="hero-inner">
<div class="hero-tag">✦ NLP · DOCUMENT-GROUNDED VERIFICATION</div>
<div class="hero-title"><span class="shield">🛡️</span> Hallucination <span class="grad">Detection</span> System</div>
<div class="hero-sub">Ask questions from your trusted PDF or video, and check every AI claim with evidence, NLI reasoning and a reliability score.</div>
<div class="pipeline">
<span class="pipe-step"><b>1</b>Trusted source</span><span class="pipe-arrow">→</span>
<span class="pipe-step"><b>2</b>Evidence retrieval</span><span class="pipe-arrow">→</span>
<span class="pipe-step"><b>3</b>NLI verification</span><span class="pipe-arrow">→</span>
<span class="pipe-step"><b>4</b>Reliability score</span>
</div>
</div>
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR — SETTINGS
# =========================================================

with st.sidebar:
    st.markdown("### 🛡️ How it works")
    st.markdown(
        """
1. **Trusted source** — the PDF, video or text we treat as the truth.
2. **Find evidence** — the sentences closest in meaning to each claim are picked.
3. **Check with NLI** — a model decides if that evidence *supports*, *contradicts* or is *neutral* to the claim.
4. **Score** — Supported = 1, Partially = 0.5, everything else = 0 → reliability %.
"""
    )

# Fixed settings (no longer shown in the UI)
whisper_size = "base"

cfg = {
    "method": METHOD_HYBRID,
    "top_k": 3,
    "t_sup": 0.60,
    "t_part": 0.25,
    "t_con": 0.60,
    "sim_floor": 0.30,
    "sim_sup": 0.55,
    "sim_part": 0.35,
}

# =========================================================
# STEP 1 — TRUSTED SOURCE
# =========================================================

units = []
source_label = ""

with glass("card_source"):
    st.markdown(
        '<div class="step-title"><span class="step-num">1</span> Choose trusted source</div>',
        unsafe_allow_html=True,
    )

    source_type = st.radio(
        "Trusted information comes from:",
        ["📄 PDF", "🎥 Video", "📝 Text"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source_type == "📄 PDF":
        pdf_files = st.file_uploader(
            "Upload one or more PDFs with trusted information",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_upload",
        )
        total_pages = 0
        for f in pdf_files or []:
            f_units, pages = pdf_units(f.name, f.getvalue())
            units.extend(f_units)
            total_pages += pages
        if pdf_files:
            source_label = ", ".join(f.name for f in pdf_files)
            if units:
                st.success(
                    f"Loaded {len(pdf_files)} file(s) · {total_pages} page(s) · "
                    f"{len(units)} evidence sentences"
                )
            else:
                st.warning("Could not read any text. Please upload a text-based PDF (not scanned).")

    elif source_type == "🎥 Video":
        video = st.file_uploader(
            "Upload a video with trusted speech",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            key="video_upload",
        )
        if video is not None:
            video_key = f"{video.name}_{video.size}_{whisper_size}"

            if st.button("🎙️ Transcribe video"):
                with st.spinner("Transcribing… this can take a few minutes."):
                    try:
                        v_units, lang = transcribe_video(video, whisper_size)
                        st.session_state["video_cache"] = {
                            "key": video_key,
                            "units": v_units,
                            "lang": lang,
                        }
                    except Exception as error:
                        st.error(f"Video transcription failed: {error}")

            cache = st.session_state.get("video_cache")
            if cache and cache["key"] == video_key:
                units = cache["units"]
                source_label = video.name
                st.success(
                    f"Transcribed · language: {cache['lang']} · {len(units)} evidence sentences"
                )
                with st.expander("📜 View transcript with timestamps"):
                    st.text("\n".join(f"[{u['loc']}] {u['text']}" for u in units))

    else:
        pasted = st.text_area(
            "Paste trusted text",
            height=180,
            placeholder="Paste the reference text that should be treated as the truth…",
        )
        units = [
            {"text": s, "source": "Pasted text", "loc": ""}
            for s in split_sentences(pasted, min_len=15)
        ]
        source_label = "Pasted text"
        if units:
            st.success(f"{len(units)} evidence sentences ready")

    if units:
        st.markdown(source_panel(units), unsafe_allow_html=True)

# =========================================================
# STEP 2 — CLAIMS
# =========================================================


def load_demo_claims():
    st.session_state.pop("ai_meta", None)
    st.session_state["question_input"] = "What is artificial intelligence?"
    st.session_state["claims_input"] = (
        "AI systems can recognize images.\n"
        "AI was developed by Alan Turing.\n"
        "Artificial intelligence will replace all humans.\n"
        "AI is used in medicine to support diagnosis."
    )


def clear_claims():
    st.session_state.pop("ai_meta", None)
    st.session_state["question_input"] = ""
    st.session_state["claims_input"] = ""


with glass("card_ask"):
    st.markdown(
        '<div class="step-title"><span class="step-num">2</span> Ask questions and / or enter claims to verify</div>',
        unsafe_allow_html=True,
    )

    question_input = st.text_area(
        "Question(s) — answered from the trusted source (optional, one per line)",
        key="question_input",
        height=90,
        placeholder="Example:\nWhat is artificial intelligence?",
    )

    # --- Gemini: write an answer to the question, which is then verified below ---
    g1, g2 = st.columns([1.4, 3])
    generate_clicked = g1.button("🤖 Generate answer with Gemini")
    g2.caption(
        "Gemini answers from its own knowledge (it never sees your source). "
        "Then this system checks that answer against your trusted source."
    )

    if generate_clicked:
        gen_questions = [q.strip() for q in question_input.splitlines() if q.strip()][:3]
        api_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
        used_so_far = st.session_state.get("gen_count", 0)

        if not gen_questions:
            st.warning("Type a question first, then click the button.")
        elif not HAS_GENAI:
            st.warning("The google-genai package is not installed (pip install google-genai).")
        elif not api_key:
            st.info(
                "Gemini key not set. Local: put GEMINI_API_KEY=your-key in a .env file. "
                "Cloud: add GEMINI_API_KEY in the app's Secrets. "
                "You can still paste an answer manually below."
            )
        elif used_so_far + len(gen_questions) > GEN_LIMIT:
            st.warning(f"Limit reached: at most {GEN_LIMIT} Gemini answers per session. Paste an answer manually.")
        else:
            try:
                with st.spinner("Gemini is writing an answer…"):
                    parts, used_model = [], ""
                    for q in gen_questions:
                        text, used_model = generate_answer(q, api_key)
                        st.session_state["gen_count"] = st.session_state.get("gen_count", 0) + 1
                        parts.append(clean_generated(text))
                answer_text = "\n".join(p for p in parts if p)
                if answer_text:
                    # allowed: the answer box below has not been created yet in this run
                    st.session_state["claims_input"] = answer_text
                    st.session_state["ai_meta"] = {"model": used_model}
                else:
                    st.warning("Gemini's reply had no usable sentences. Try again.")
            except Exception as err:
                st.error(friendly_error(err))

    claims_input = st.text_area(
        "AI answer / claims to verify — one claim per line",
        key="claims_input",
        height=150,
        placeholder="Example:\nArtificial intelligence is a field of computer science.\nAI can recognize images.",
    )
    if st.session_state.get("ai_meta"):
        st.caption(f"🤖 Answer generated by {st.session_state['ai_meta']['model']} — you can edit it before verifying.")

    claims, skipped_questions = extract_claims(claims_input)

    # Questions typed in the Question box, plus any question lines typed in the answer box
    typed_questions = [q.strip() for q in question_input.splitlines() if q.strip()]
    questions = list(dict.fromkeys(typed_questions + skipped_questions))

    # With exactly one question it is also used as extra context when searching evidence for claims
    question_used = questions[0] if len(questions) == 1 else ""

    c1, c2, c3 = st.columns([1, 1, 4])
    c1.button("✨ Demo claims", on_click=load_demo_claims)
    c2.button("🗑️ Clear", on_click=clear_claims)
    c3.caption(f"{len(claims)} claim(s) · {len(questions)} question(s) detected")

    if skipped_questions:
        st.info(
            f"{len(skipped_questions)} question(s) found in the answer box — "
            "they will be answered from the source instead of being checked as claims."
        )

analyze = st.button("🔍 Get answers & verify claims", type="primary", use_container_width=True)

# =========================================================
# RUN ANALYSIS
# =========================================================

if analyze:
    if not units:
        st.warning("Please provide a trusted source first (PDF, transcribed video or text).")
    elif not claims and not questions:
        st.warning("Please enter a question, some claims, or both.")
    else:
        with st.status("Running verification pipeline…", expanded=True) as status:
            st.write("Embedding the trusted source…")
            texts = tuple(u["text"] for u in units)
            embeddings = embed_texts(texts)
            index = build_index(embeddings)
            vocab = build_vocab(texts)

            answers = []
            if questions:
                st.write("Finding answers to your question(s) in the source…")
                answers = [
                    {"question": q, **answer_question(q, units, index)} for q in questions
                ]

            rows = []
            if claims:
                st.write("Loading the NLI model (the first run downloads a few hundred MB — please wait)…")
                load_nli_model()

                st.write("Verifying claims…")
                progress = st.progress(0.0)
                for n, claim in enumerate(claims, start=1):
                    rows.append(
                        verify_claim(n, claim, question_used, units, index, vocab, cfg)
                    )
                    progress.progress(n / len(claims))

            status.update(label="Analysis complete", state="complete", expanded=False)

        statuses = [r["status"] for r in rows]
        score = reliability_score(statuses)
        stamp = datetime.now()

        st.session_state["analysis"] = {
            "id": stamp.strftime("%Y%m%d%H%M%S"),
            "time": stamp.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source_label,
            "n_evidence": len(units),
            "question": question_used,
            "answers": answers,
            "generated_by": (st.session_state.get("ai_meta") or {}).get("model", ""),
            "cfg": dict(cfg),
            "rows": rows,
        }

# =========================================================
# RESULTS
# =========================================================

analysis = st.session_state.get("analysis")

if analysis:
    rows = analysis["rows"]

    st.markdown("## 📊 Results")
    st.caption(
        f"Source: {analysis['source']} · {analysis['n_evidence']} evidence sentences · "
        f"{analysis['time']}"
        + (f" · answer written by {analysis['generated_by']}" if analysis.get("generated_by") else "")
    )

    for a in analysis.get("answers", []):
        st.markdown(answer_card(a), unsafe_allow_html=True)
        if a["found"]:
            with st.expander("📚 Sentences used for this answer"):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Source": h["citation"],
                                "Relevance": round(h["sim"], 2),
                                "Sentence": h["text"],
                            }
                            for h in a["hits"]
                        ]
                    ),
                    hide_index=True,
                )

    if not rows:
        st.stop()

    total = len(rows)
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in ALL_STATUSES}
    score = reliability_score([r["status"] for r in rows])
    score_pct = round(score * 100, 1)
    halluc_rate = round((counts[S_CON] + counts[S_UNS]) / total * 100, 1)

    if score_pct >= 80:
        ring_color, verdict_fn, verdict = "#22c55e", st.success, "High reliability — most claims are supported by the trusted source."
    elif score_pct >= 50:
        ring_color, verdict_fn, verdict = "#eab308", st.warning, "Moderate reliability — some claims are unsupported or only partly supported."
    else:
        ring_color, verdict_fn, verdict = "#ef4444", st.error, "Low reliability — a large share of the claims is unsupported or contradicted."

    tab_overview, tab_claims, tab_compare, tab_export = st.tabs(
        ["📊 Overview", "🔎 Claim details", "⚖️ Method comparison", "📥 Export"]
    )

    # ---------------- OVERVIEW ----------------
    with tab_overview:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.markdown(kpi(total, "Total claims"), unsafe_allow_html=True)
        k2.markdown(kpi(counts[S_SUP], "✅ Supported", STATUS_META[S_SUP]["color"]), unsafe_allow_html=True)
        k3.markdown(kpi(counts[S_PART], "⚠️ Partially", STATUS_META[S_PART]["color"]), unsafe_allow_html=True)
        k4.markdown(kpi(counts[S_CON], "⛔ Contradicted", STATUS_META[S_CON]["color"]), unsafe_allow_html=True)
        k5.markdown(kpi(counts[S_UNS], "❓ Unsupported", STATUS_META[S_UNS]["color"]), unsafe_allow_html=True)

        st.write("")
        left, right = st.columns([1, 1])

        with left:
            st.markdown(ring(score_pct, ring_color), unsafe_allow_html=True)
            st.markdown(
                f"<div style='text-align:center;opacity:.8'>Hallucination rate: "
                f"<b>{halluc_rate}%</b> (contradicted + unsupported)</div>",
                unsafe_allow_html=True,
            )

        with right:
            counts_df = pd.DataFrame(
                {"Status": ALL_STATUSES, "Count": [counts[s] for s in ALL_STATUSES]}
            )
            counts_df = counts_df[counts_df["Count"] > 0]
            donut = (
                alt.Chart(counts_df)
                .mark_arc(innerRadius=55, outerRadius=95)
                .encode(
                    theta=alt.Theta("Count:Q"),
                    color=alt.Color(
                        "Status:N",
                        scale=alt.Scale(
                            domain=ALL_STATUSES,
                            range=[STATUS_META[s]["color"] for s in ALL_STATUSES],
                        ),
                        legend=alt.Legend(orient="bottom", title=None),
                    ),
                    tooltip=["Status", "Count"],
                )
                .properties(height=250)
                .configure(background="rgba(0,0,0,0)")
                .configure_view(stroke=None)
            )
            st.altair_chart(donut)

        verdict_fn(verdict)
        st.caption(
            "Reliability = (Supported × 1.0 + Partially Supported × 0.5 + "
            "Contradicted × 0 + Unsupported × 0) ÷ total claims."
        )
        st.caption(
            "Unsupported means the claim was not found in the trusted source. "
            "It can still be true in the real world; this system checks faithfulness to the source."
        )

        st.markdown("#### Claim-by-claim result")
        for r in rows:
            st.markdown(summary_row(r), unsafe_allow_html=True)

        st.markdown("#### NLI probabilities per claim")
        long_rows = []
        for r in rows:
            for name, key in [("Entailment", "entail"), ("Neutral", "neutral"), ("Contradiction", "contra")]:
                long_rows.append({"Claim": str(r["id"]), "Type": name, "Probability": r[key]})
        bars = (
            alt.Chart(pd.DataFrame(long_rows))
            .mark_bar()
            .encode(
                x=alt.X("Claim:N", sort=None, title="Claim number", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Probability:Q", axis=alt.Axis(format="%")),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(
                        domain=["Entailment", "Neutral", "Contradiction"],
                        range=["#22c55e", "#94a3b8", "#ef4444"],
                    ),
                    legend=alt.Legend(orient="bottom", title=None),
                ),
                tooltip=["Claim", "Type", alt.Tooltip("Probability:Q", format=".0%")],
            )
            .properties(height=260, width="container")
            .configure(background="rgba(0,0,0,0)")
            .configure_view(stroke=None)
        )
        st.altair_chart(bars)

    # ---------------- CLAIM DETAILS ----------------
    with tab_claims:
        wanted = st.multiselect("Filter by status", ALL_STATUSES, default=ALL_STATUSES)
        for r in rows:
            if r["status"] not in wanted:
                continue
            st.markdown(claim_card(r), unsafe_allow_html=True)
            with st.expander(f"📚 Evidence for claim {r['id']}"):
                st.markdown(f"**Full claim {r['id']}:** {r['claim']}")
                st.markdown(f"**Best evidence** — {r['citation']}")
                st.markdown(
                    f'<div class="ev-box">{highlight_evidence(r["evidence"], r["claim"])}</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Highlighted words also appear in the claim.")
                st.markdown(f"Retrieval similarity: **{r['sim']:.2f}**")
                st.markdown("**Top retrieved sentences**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Source": c["citation"],
                                "Similarity": round(c["sim"], 2),
                                "Entail.": round(c["entail"], 2),
                                "Neutral": round(c["neutral"], 2),
                                "Contra.": round(c["contra"], 2),
                                "Sentence": c["text"],
                            }
                            for c in r["candidates"]
                        ]
                    ),
                    hide_index=True,
                )

    # ---------------- METHOD COMPARISON ----------------
    with tab_compare:
        st.markdown(
            "Compare **similarity-only** (baseline) with **NLI** and the **hybrid** approach. "
            "Fill the *Expected* column with the correct label to measure accuracy."
        )

        base_df = pd.DataFrame(
            {
                "Claim": [r["claim"] for r in rows],
                "Expected": [None] * total,
                "Similarity only": [r["status_sim"] for r in rows],
                "NLI only": [r["status_nli"] for r in rows],
                "Hybrid": [r["status_hybrid"] for r in rows],
            }
        )

        edited = st.data_editor(
            base_df,
            key=f"eval_{analysis['id']}",
            hide_index=True,
            disabled=["Claim", "Similarity only", "NLI only", "Hybrid"],
            column_config={
                "Expected": st.column_config.SelectboxColumn(
                    "Expected (your label)", options=ALL_STATUSES
                ),
            },
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Reliability · Similarity only", f"{reliability_score(base_df['Similarity only']) * 100:.1f}%")
        m2.metric("Reliability · NLI only", f"{reliability_score(base_df['NLI only']) * 100:.1f}%")
        m3.metric("Reliability · Hybrid", f"{reliability_score(base_df['Hybrid']) * 100:.1f}%")

        labelled = edited.dropna(subset=["Expected"])
        if labelled.empty:
            st.info("Choose an *Expected* label for at least one claim to see accuracy.")
        else:
            st.markdown(f"#### Accuracy on {len(labelled)} labelled claim(s)")

            def is_hallucination(label):
                return label in (S_CON, S_UNS)

            exp_bin = labelled["Expected"].map(is_hallucination)
            acc_rows = []
            for col in ["Similarity only", "NLI only", "Hybrid"]:
                exact = (labelled[col] == labelled["Expected"]).mean()
                binary = (labelled[col].map(is_hallucination) == exp_bin).mean()
                acc_rows.append(
                    {
                        "Method": col,
                        "Exact label accuracy": f"{exact:.0%}",
                        "Hallucination detected correctly": f"{binary:.0%}",
                    }
                )
            st.dataframe(pd.DataFrame(acc_rows), hide_index=True)
            st.caption(
                "“Hallucination detected correctly” treats Contradicted and Unsupported as "
                "hallucinations and the other two labels as faithful."
            )

    # ---------------- EXPORT ----------------
    with tab_export:
        report_df = pd.DataFrame(
            [
                {
                    "claim_no": r["id"],
                    "question": analysis.get("question", ""),
                    "claim": r["claim"],
                    "status": r["status"],
                    "similarity": round(r["sim"], 4),
                    "entailment": round(r["entail"], 4),
                    "neutral": round(r["neutral"], 4),
                    "contradiction": round(r["contra"], 4),
                    "evidence": r["evidence"],
                    "citation": r["citation"],
                    "explanation": r["explanation"],
                    "missing_terms": ", ".join(r["missing"]),
                }
                for r in rows
            ]
        )
        payload = {
            "generated": analysis["time"],
            "source": analysis["source"],
            "question": analysis.get("question", ""),
            "answers": analysis.get("answers", []),
            "settings": analysis["cfg"],
            "reliability_percent": score_pct,
            "hallucination_rate_percent": halluc_rate,
            "counts": counts,
            "claims": rows,
        }

        d1, d2, d3 = st.columns(3)
        d1.download_button(
            "⬇️ Download CSV report",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="hallucination_report.csv",
            mime="text/csv",
        )
        d2.download_button(
            "⬇️ Download JSON report",
            data=json.dumps(payload, indent=2, default=str).encode("utf-8"),
            file_name="hallucination_report.json",
            mime="application/json",
        )
        d3.download_button(
            "⬇️ Printable report (HTML)",
            data=build_html_report(analysis, counts, score_pct, halluc_rate).encode("utf-8"),
            file_name="hallucination_report.html",
            mime="text/html",
        )
        st.caption("Open the HTML report in a browser and use Print → Save as PDF for a ready-made PDF.")
        st.dataframe(report_df, hide_index=True)

# =========================================================
# FOOTER + SIDEBAR HISTORY
# =========================================================

st.markdown(
    '<div class="footer">NLP-Based Hallucination Detection and Reliability Scoring System · Prototype</div>',
    unsafe_allow_html=True,
)
