# NLP-Based Hallucination Detection and Reliability Scoring System

A document-grounded system that checks whether the claims in an AI-generated answer are supported by a **trusted source** (PDF, video or pasted text), and gives every claim a label, an explanation and evidence with its location.

## What it does

1. **Trusted source** – upload PDF(s), upload a video (speech is transcribed with timestamps) or paste text.
2. **Question answering** – questions are answered from the source, with page / timestamp citations.
3. **Claim verification** – every claim is checked against the most relevant evidence sentences.
4. **Reliability score** – the share of supported claims, plus a hallucination rate.

### Claim labels

| Label | Meaning |
|---|---|
| Supported | The source says the same thing (NLI entailment) |
| Partially Supported | The source backs part of the claim |
| Contradicted | The source says the opposite (NLI contradiction) |
| Unsupported | The source neither confirms nor denies the claim (NLI neutral / not found) |

**Reliability** = (Supported × 1.0 + Partially Supported × 0.5 + Contradicted × 0 + Unsupported × 0) ÷ total claims.

## Pipeline

```
Trusted source (PDF / video / text)
  → sentence-level evidence units (with page / timestamp)
  → Sentence-Transformer embeddings → top-k evidence retrieval
  → NLI model (entailment / neutral / contradiction)
  → claim label + explanation → reliability score → report (CSV / JSON / HTML)
```

## Models (all open source, run on CPU)

| Task | Model |
|---|---|
| Evidence retrieval | `sentence-transformers/all-MiniLM-L6-v2` |
| Claim verification (NLI) | `cross-encoder/nli-MiniLM2-L6-H768` |
| Speech-to-text | `faster-whisper` (`base`, int8) |

The app also compares three methods side by side: similarity only, NLI only and the hybrid approach.

## Run locally

Python 3.11 or 3.12 is recommended.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

The models are downloaded automatically on first use.

**Optional:** put an image named `background.jpg` next to `app.py` to use it as the page background.

## Optional: generate the answer with Gemini (Google GenAI)

Type a question and click **Generate answer with Gemini**. Gemini answers from its own knowledge (it never sees your source documents; only the question is sent). The answer is then split into claims and verified against the trusted source, so you can measure how faithful an LLM answer is to a document.

1. Create a free API key in Google AI Studio.
2. **Local:** copy `.env.example` to `.env` and put the key in it.
3. **Streamlit Cloud:** App settings → Secrets → `GEMINI_API_KEY = "your-key"`.

Never commit the real key to GitHub. At most 8 Gemini answers are allowed per browser session to protect the free quota.

Note: *Unsupported* means "not found in the trusted source", not "false in the real world".

## Limitations / future work

- Claims are checked sentence by sentence; compound claims are not split yet.
- NLI works best on short, self-contained sentences.
- Planned: claim splitting, larger NLI model, evaluation on a labelled test set (accuracy, precision, recall, F1, confusion matrix).
