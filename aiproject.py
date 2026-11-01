
"""
AI Assessment Generator & Comparative Evaluation Arena
------------------------------------------------------
Streamlit + Google Gemini (google-genai) + Hugging Face Inference + SQLite3.
 
Run:  streamlit run app.py
"""
 
import io
import json
import sqlite3
import time
import uuid
from datetime import datetime
 
import pandas as pd
import streamlit as st
 
# ----------------------------------------------------------------------------
# HARDCODED CONFIGURATION (edit these constants - never asked in the UI)
# ----------------------------------------------------------------------------
GEMINI_API_KEY = ""
HF_TOKEN = ""
 
GEMINI_MODEL = "gemini-3.8-flash"
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # or "meta-llama/Llama-3-8B-Instruct"
 
DB_PATH = "quiz_analytics.db"
MAX_PDF_PAGES = 10
MAX_SOURCE_CHARS = 20_000
 
MODEL_LABEL_GEMINI = "Google Gemini"
MODEL_LABEL_HF = "Hugging Face Mistral-7B"
 
ENGINE_GEMINI_OPT = f"Google Gemini ({GEMINI_MODEL})"
ENGINE_HF_OPT = "Hugging Face Open-Source (Mistral-7B-Instruct)"
 
TYPE_MCQ = "Multiple Choice Questions (MCQs)"
TYPE_SA = "Short-Answer Questions"
DIFFICULTIES = [
    "Easy (Recall/Definitions)",
    "Medium (Application/Understanding)",
    "Hard (Analytical/Bloom's Taxonomy)",
]
 
BLOOM_DEFS = {
    DIFFICULTIES[0]: (
        "Bloom's Tier: RECALL (Remember). Questions test retrieval of facts, "
        "terms, definitions and basic concepts directly stated in the text."
    ),
    DIFFICULTIES[1]: (
        "Bloom's Tier: APPLICATION (Understand/Apply). Questions require "
        "explaining ideas in one's own words or applying concepts to new, "
        "concrete scenarios."
    ),
    DIFFICULTIES[2]: (
        "Bloom's Tier: ANALYSIS (Analyze/Evaluate). Questions require breaking "
        "information into parts, comparing, finding cause-effect relations, "
        "or critically justifying a position."
    ),
}
 
 
# ----------------------------------------------------------------------------
# DATABASE LAYER
# ----------------------------------------------------------------------------
def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)
 
 
def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS assessment_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                model_used TEXT,
                topic_snippet TEXT,
                assessment_type TEXT,
                difficulty TEXT,
                score INTEGER,
                total_questions INTEGER,
                percentage REAL,
                latency_seconds REAL
            )
            """
        )
        conn.commit()
 
 
def save_assessment_result(
    model_used: str,
    topic_snippet: str,
    assessment_type: str,
    difficulty: str,
    score: int,
    total_questions: int,
    latency_seconds: float,
) -> None:
    pct = round((score / total_questions) * 100, 2) if total_questions else 0.0
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO assessment_records
            (timestamp, model_used, topic_snippet, assessment_type, difficulty,
             score, total_questions, percentage, latency_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_used,
                topic_snippet[:60],
                assessment_type,
                difficulty,
                int(score),
                int(total_questions),
                pct,
                round(float(latency_seconds), 3),
            ),
        )
        conn.commit()
 
 
def fetch_assessment_history(limit: int = 500) -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql_query(
            "SELECT * FROM assessment_records ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
 
 
# ----------------------------------------------------------------------------
# PDF / TEXT HANDLING
# ----------------------------------------------------------------------------
def extract_pdf_text(uploaded_file) -> str:
    """Bounded extraction: max MAX_PDF_PAGES pages, MAX_SOURCE_CHARS chars."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
 
    try:
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    except PdfReadError as e:
        raise ValueError(f"Could not read this PDF: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Invalid or corrupted PDF file: {e}") from e
 
    if reader.is_encrypted:
        try:
            if reader.decrypt("") == 0:
                raise ValueError("This PDF is password-protected. Please upload an unlocked copy.")
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ValueError("This PDF is encrypted and could not be opened.") from e
 
    chunks, total = [], 0
    for i, page in enumerate(reader.pages):
        if i >= MAX_PDF_PAGES:
            break
        try:
            text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - skip unreadable pages
            continue
        text = text.strip()
        if not text:
            continue
        chunks.append(text)
        total += len(text)
        if total >= MAX_SOURCE_CHARS:
            break
 
    combined = "\n\n".join(chunks)[:MAX_SOURCE_CHARS]
    if not combined.strip():
        raise ValueError("No extractable text found (the PDF may be scanned or image-only).")
    return combined
 
 
# ----------------------------------------------------------------------------
# PROMPT + LLM CALLS
# ----------------------------------------------------------------------------
def build_prompt(source: str, a_type: str, difficulty: str, n: int) -> str:
    bloom = BLOOM_DEFS[difficulty]
    if a_type == TYPE_MCQ:
        schema = (
            f"Return a JSON array of exactly {n} objects. Each object MUST have exactly these keys:\n"
            '  "question": string,\n'
            '  "options": array of exactly 4 UNIQUE strings (no "A)" / "1." prefixes),\n'
            '  "correct_answer": string that is IDENTICAL to one of the options,\n'
            '  "explanation": string of 1-2 punchy sentences.'
        )
    else:
        schema = (
            f"Return a JSON array of exactly {n} objects. Each object MUST have exactly these keys:\n"
            '  "question": string,\n'
            '  "model_answer": string of 1-3 concise sentences,\n'
            '  "rubric_keywords": array of 3-6 short keyword strings.'
        )
    return f"""You are an expert educational assessment designer.
Create {n} {a_type} based ONLY on the study material below.
 
{bloom}
Difficulty level: {difficulty}
 
OUTPUT CONTRACT:
{schema}
- Output ONLY the raw JSON array. No markdown fences, no commentary, no preface.
- Keep every explanation/answer to 1-2 punchy sentences.
 
STUDY MATERIAL:
\"\"\"
{source}
\"\"\"
"""
 
 
def _keys_missing(engine: str) -> str | None:
    if engine == "gemini" and (not GEMINI_API_KEY or GEMINI_API_KEY.startswith("YOUR_")):
        return "GEMINI_API_KEY is missing or still a placeholder. Edit the constant at the top of app.py."
    if engine == "hf" and (not HF_TOKEN or HF_TOKEN.startswith("YOUR_")):
        return "HF_TOKEN is missing or still a placeholder. Edit the constant at the top of app.py."
    return None
 
 
def call_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types
 
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            response_mime_type="application/json",
        ),
    )
    return resp.text or ""
 
 
def call_hf(prompt: str) -> str:
    from huggingface_hub import InferenceClient
 
    client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)
    resp = client.chat_completion(
        messages=[
            {"role": "system", "content": "You output only valid JSON arrays."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2000,
        temperature=0.4,
    )
    return resp.choices[0].message.content or ""
 
 
def parse_json_array(raw: str) -> list:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON array.")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, list) or not data:
        raise ValueError("Model returned an empty or non-array JSON payload.")
    return data
 
 
def validate_questions(data: list, a_type: str) -> list:
    cleaned = []
    for item in data:
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            continue
        if a_type == TYPE_MCQ:
            opts = item.get("options")
            if not isinstance(opts, list):
                continue
            opts = list(dict.fromkeys(str(o).strip() for o in opts if str(o).strip()))
            if len(opts) < 2:
                continue
            ans = str(item.get("correct_answer", "")).strip()
            if ans not in opts:
                # tolerate letter answers ("B") or case/whitespace mismatches
                lower = {o.lower(): o for o in opts}
                if ans.lower() in lower:
                    ans = lower[ans.lower()]
                elif len(ans) == 1 and ans.upper() in "ABCD"[: len(opts)]:
                    ans = opts[ "ABCD".index(ans.upper()) ]
                else:
                    continue
            cleaned.append(
                {
                    "question": str(item["question"]).strip(),
                    "options": opts,
                    "correct_answer": ans,
                    "explanation": str(item.get("explanation", "")).strip(),
                }
            )
        else:
            kws = item.get("rubric_keywords", [])
            if not isinstance(kws, list):
                kws = [str(kws)]
            cleaned.append(
                {
                    "question": str(item["question"]).strip(),
                    "model_answer": str(item.get("model_answer", "")).strip(),
                    "rubric_keywords": [str(k).strip() for k in kws if str(k).strip()],
                }
            )
    if not cleaned:
        raise ValueError("None of the generated questions matched the required schema.")
    return cleaned
 
 
def generate_assessment(source: str, engine: str, a_type: str, difficulty: str, n: int):
    """Returns (questions, latency_seconds). Raises on failure."""
    problem = _keys_missing(engine)
    if problem:
        raise RuntimeError(problem)
    prompt = build_prompt(source, a_type, difficulty, n)
    t0 = time.perf_counter()
    raw = call_gemini(prompt) if engine == "gemini" else call_hf(prompt)
    latency = time.perf_counter() - t0
    questions = validate_questions(parse_json_array(raw), a_type)
    return questions, latency
 
 
# ----------------------------------------------------------------------------
# EXPORT HELPERS
# ----------------------------------------------------------------------------
def export_markdown(meta: dict, questions: list) -> str:
    lines = [
        "# Generated Assessment",
        f"- **Model:** {meta['model_label']}",
        f"- **Type:** {meta['a_type']}",
        f"- **Difficulty:** {meta['difficulty']}",
        f"- **Generated:** {meta['created']}",
        "",
    ]
    for i, q in enumerate(questions, 1):
        lines.append(f"## Q{i}. {q['question']}")
        if meta["a_type"] == TYPE_MCQ:
            for j, o in enumerate(q["options"]):
                lines.append(f"- {chr(65 + j)}. {o}")
            lines.append(f"\n**Answer:** {q['correct_answer']}")
            lines.append(f"\n**Explanation:** {q['explanation']}\n")
        else:
            lines.append(f"\n**Model Answer:** {q['model_answer']}")
            lines.append(f"\n**Rubric Keywords:** {', '.join(q['rubric_keywords'])}\n")
    return "\n".join(lines)
 
 
def export_text(meta: dict, questions: list) -> str:
    lines = [
        "GENERATED ASSESSMENT",
        f"Model: {meta['model_label']} | Type: {meta['a_type']} | Difficulty: {meta['difficulty']}",
        f"Generated: {meta['created']}",
        "=" * 60,
    ]
    for i, q in enumerate(questions, 1):
        lines.append(f"\nQ{i}. {q['question']}")
        if meta["a_type"] == TYPE_MCQ:
            for j, o in enumerate(q["options"]):
                lines.append(f"   {chr(65 + j)}) {o}")
            lines.append(f"   Answer: {q['correct_answer']}")
            lines.append(f"   Explanation: {q['explanation']}")
        else:
            lines.append(f"   Model Answer: {q['model_answer']}")
            lines.append(f"   Rubric Keywords: {', '.join(q['rubric_keywords'])}")
    return "\n".join(lines)
 
 
# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("⚙️ Configuration")
    engine_choice = st.sidebar.radio("Model Engine", [ENGINE_GEMINI_OPT, ENGINE_HF_OPT])
    engine = "gemini" if engine_choice == ENGINE_GEMINI_OPT else "hf"
 
    st.sidebar.subheader("Assessment Parameters")
    a_type = st.sidebar.selectbox("Assessment Type", [TYPE_MCQ, TYPE_SA])
    difficulty = st.sidebar.selectbox("Difficulty", DIFFICULTIES)
    n = st.sidebar.slider("Number of Questions", 3, 10, 5)
 
    problem = _keys_missing(engine)
    if problem:
        st.sidebar.error("🔴 Engine not ready: API key not configured in app.py")
    else:
        st.sidebar.success("🟢 Active connection ready")
    return engine, a_type, difficulty, n
 
 
def render_generator_tab(engine, a_type, difficulty, n):
    st.subheader("1️⃣ Provide Study Material")
    source_text = ""
    t_paste, t_pdf = st.tabs(["📋 Paste Text", "📄 Upload PDF"])
    with t_paste:
        pasted = st.text_area(
            "Paste your study text",
            height=220,
            placeholder=f"Only the first {MAX_SOURCE_CHARS:,} characters are used.",
        )
    with t_pdf:
        pdf = st.file_uploader(
            f"Upload a PDF (first {MAX_PDF_PAGES} pages only)", type=["pdf"]
        )
 
    if st.button("✨ Generate Assessment", type="primary"):
        try:
            if pdf is not None:
                with st.spinner("Extracting PDF text..."):
                    source_text = extract_pdf_text(pdf)
            else:
                source_text = pasted.strip()[:MAX_SOURCE_CHARS]
            if len(source_text) < 50:
                st.warning("Please provide at least a few sentences of study material.")
            else:
                label = MODEL_LABEL_GEMINI if engine == "gemini" else MODEL_LABEL_HF
                with st.spinner(f"Generating with {label}..."):
                    questions, latency = generate_assessment(
                        source_text, engine, a_type, difficulty, n
                    )
                st.session_state["assessment"] = {
                    "id": uuid.uuid4().hex[:8],
                    "questions": questions,
                    "latency": latency,
                    "saved": False,
                    "results": None,
                    "meta": {
                        "model_label": label,
                        "a_type": a_type,
                        "difficulty": difficulty,
                        "topic": source_text[:60].replace("\n", " "),
                        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    },
                }
        except json.JSONDecodeError:
            st.error("The model returned malformed JSON. Please try again.")
        except ValueError as e:
            st.error(f"⚠️ {e}")
        except RuntimeError as e:
            st.error(f"🔑 {e}")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if any(k in msg.lower() for k in ("api key", "401", "403", "unauthorized", "permission")):
                st.error("Authentication failed. Check the hardcoded key constants in app.py.")
            else:
                st.error(f"Generation failed: {msg}")
 
    data = st.session_state.get("assessment")
    if not data:
        return
 
    meta, questions = data["meta"], data["questions"]
    st.divider()
    st.subheader("2️⃣ Test Arena")
    st.caption(
        f"{meta['model_label']} • {meta['a_type']} • {meta['difficulty']} • "
        f"⏱️ generated in {data['latency']:.2f}s"
    )
 
    if meta["a_type"] == TYPE_MCQ:
        render_mcq(data)
    else:
        for i, q in enumerate(questions, 1):
            with st.expander(f"Q{i}. {q['question']}"):
                st.markdown(f"**Model answer:** {q['model_answer']}")
                st.markdown("**Rubric keywords:** " + (", ".join(f"`{k}`" for k in q["rubric_keywords"]) or "—"))
 
    st.subheader("3️⃣ Export")
    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇️ Download Markdown (.md)",
        export_markdown(meta, questions),
        file_name="assessment.md",
        mime="text/markdown",
    )
    c2.download_button(
        "⬇️ Download Text (.txt)",
        export_text(meta, questions),
        file_name="assessment.txt",
        mime="text/plain",
    )
 
 
def render_mcq(data):
    questions, meta = data["questions"], data["meta"]
    aid = data["id"]
 
    with st.form(f"quiz_form_{aid}"):
        for i, q in enumerate(questions):
            st.markdown(f"**Q{i + 1}. {q['question']}**")
            st.radio(
                "Select an answer",
                q["options"],
                index=None,
                key=f"ans_{aid}_{i}",
                label_visibility="collapsed",
            )
            st.write("")
        submitted = st.form_submit_button("✅ Submit Answers")
 
    if submitted:
        picks = [st.session_state.get(f"ans_{aid}_{i}") for i in range(len(questions))]
        score = sum(1 for p, q in zip(picks, questions) if p == q["correct_answer"])
        data["results"] = {"picks": picks, "score": score}
        if not data["saved"]:
            try:
                save_assessment_result(
                    model_used=meta["model_label"],
                    topic_snippet=meta["topic"],
                    assessment_type="MCQ",
                    difficulty=meta["difficulty"],
                    score=score,
                    total_questions=len(questions),
                    latency_seconds=data["latency"],
                )
                data["saved"] = True
            except sqlite3.Error as e:
                st.error(f"Could not save result to database: {e}")
 
    res = data.get("results")
    if res:
        total = len(questions)
        pct = res["score"] / total if total else 0
        st.markdown(f"### 🎯 Score: {res['score']}/{total} ({pct * 100:.0f}%)")
        st.progress(pct)
        if data["saved"]:
            st.caption("Saved to analytics history.")
        for i, (pick, q) in enumerate(zip(res["picks"], questions), 1):
            if pick == q["correct_answer"]:
                st.success(f"**Q{i}: Correct ✅** — {q['correct_answer']}\n\n💡 {q['explanation']}")
            else:
                shown = pick if pick is not None else "No answer"
                st.error(
                    f"**Q{i}: Incorrect ❌** — Your answer: {shown}\n\n"
                    f"Correct: **{q['correct_answer']}**\n\n💡 {q['explanation']}"
                )
 
 
def render_analytics_tab():
    try:
        df = fetch_assessment_history()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return
 
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Quizzes Taken", len(df))
    c2.metric("Average Score %", f"{df['percentage'].mean():.1f}%" if len(df) else "—")
    c3.metric("Avg Generation Latency (s)", f"{df['latency_seconds'].mean():.2f}" if len(df) else "—")
 
    if df.empty:
        st.info("No attempts yet. Take an MCQ test in the first tab to see analytics.")
        return
 
    st.subheader("Attempt History")
    view = df[["timestamp", "model_used", "score", "total_questions", "percentage", "latency_seconds"]].rename(
        columns={
            "timestamp": "Date",
            "model_used": "Model",
            "score": "Score",
            "total_questions": "Total",
            "percentage": "%",
            "latency_seconds": "Latency (s)",
        }
    )
    st.dataframe(view, use_container_width=True, hide_index=True)
 
    st.subheader("Model Comparison")
    agg = df.groupby("model_used").agg(
        avg_latency=("latency_seconds", "mean"), avg_score=("percentage", "mean")
    )
    left, right = st.columns(2)
    with left:
        st.caption("Average generation latency (s)")
        st.bar_chart(agg[["avg_latency"]])
    with right:
        st.caption("Average score (%)")
        st.bar_chart(agg[["avg_score"]])
 
 
def main():
    st.set_page_config(page_title="AI Assessment Arena", page_icon="🎓", layout="wide")
    init_db()
    st.title("🎓 AI Assessment Generator & Model Comparison Arena")
    st.caption("Gemini vs. open-source LLMs • Bloom's Taxonomy tiers • persistent analytics")
 
    engine, a_type, difficulty, n = render_sidebar()
    tab1, tab2 = st.tabs(["🚀 Assessment Generator & Test Arena", "📊 Analytics & Attempt History"])
    with tab1:
        render_generator_tab(engine, a_type, difficulty, n)
    with tab2:
        render_analytics_tab()
 
 
if __name__ == "__main__":
    main()