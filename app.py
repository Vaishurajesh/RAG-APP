"""
app.py
------
Streamlit UI for the Hasamex AI Engineer case study.

Tabs:
  1. Interview Guide       - grounded answer per expert per guide question,
                              with verified quotes + timestamps
  2. Themes & Disagreements - cross-transcript synthesis per question
  3. Ask a Question          - free-form RAG chat across all 3 transcripts
  4. Browse Transcripts      - raw transcript viewer, for manual spot-checking

Run:
    export GROQ_API_KEY=gsk_...
    pip install -r requirements.txt
    streamlit run app.py
"""
import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.parser import parse_transcript, parse_interview_guide
from src.retrieval import build_indexes, build_global_index
from src.pipeline import answer_guide_question, answer_chat_question, synthesize_theme, GroundedAnswer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRANSCRIPT_FILES = ["Transcript_1_France.txt", "Transcript_2_Germany.txt", "Transcript_3_UK.txt"]
GUIDE_FILE = "Interview_Guide.txt"

st.set_page_config(page_title="Hasamex Expert Call Analyzer", layout="wide")


@st.cache_resource(show_spinner="Parsing transcripts and building indexes...")
def load_data():
    turns_by_transcript = {}
    for fname in TRANSCRIPT_FILES:
        turns = parse_transcript(os.path.join(DATA_DIR, fname))
        tid = turns[0].transcript_id if turns else fname
        turns_by_transcript[tid] = turns
    questions = parse_interview_guide(os.path.join(DATA_DIR, GUIDE_FILE))
    per_transcript_indexes = build_indexes(turns_by_transcript)
    global_index = build_global_index(turns_by_transcript)
    return turns_by_transcript, questions, per_transcript_indexes, global_index


def render_grounded_answer(ga: GroundedAnswer):
    if ga.insufficient_evidence:
        st.warning("⚠️ Not clearly addressed in this transcript.")
    st.write(ga.answer)
    if ga.citations:
        with st.expander(f"📌 {len(ga.citations)} supporting quote(s)", expanded=False):
            for c in ga.citations:
                st.markdown(f"> \"{c.quote}\"")
                st.caption(f"— {c.expert_name} · {c.market} · {c.timestamp}")
    else:
        st.caption("No verified supporting quote found.")


def main():
    st.title("🔍 Expert Call Analyzer")
    st.caption("Robotic Surgery Market — France / Germany / UK expert interviews")

    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set. Export it in your shell before running `streamlit run app.py`.")
        st.stop()

    turns_by_transcript, questions, per_transcript_indexes, global_index = load_data()
    expert_meta = {
        tid: (turns[0].expert_name, turns[0].market) for tid, turns in turns_by_transcript.items() if turns
    }

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Interview Guide", "🧩 Themes & Disagreements", "💬 Ask a Question", "📄 Browse Transcripts"]
    )

    # ---------------- Tab 1: Interview Guide ----------------
    with tab1:
        st.subheader("Answers to the Interview Guide, per expert")
        q_choice = st.selectbox("Select a question", questions, key="guide_q")
        if st.button("Generate answers", key="guide_btn"):
            cols = st.columns(len(turns_by_transcript))
            answers_cache = {}
            for col, (tid, index) in zip(cols, per_transcript_indexes.items()):
                name, market = expert_meta[tid]
                with col:
                    st.markdown(f"**{name}** · {market}")
                    with st.spinner("Analyzing..."):
                        ga = answer_guide_question(q_choice, index)
                    render_grounded_answer(ga)
                    answers_cache[name] = ga
            st.session_state[f"cache::{q_choice}"] = answers_cache

    # ---------------- Tab 2: Themes & Disagreements ----------------
    with tab2:
        st.subheader("Cross-transcript synthesis")
        q_choice2 = st.selectbox("Select a question to synthesize", questions, key="synth_q")
        if st.button("Synthesize themes & disagreements", key="synth_btn"):
            cache_key = f"cache::{q_choice2}"
            if cache_key in st.session_state:
                per_expert_answers = st.session_state[cache_key]
            else:
                per_expert_answers = {}
                for tid, index in per_transcript_indexes.items():
                    name, _ = expert_meta[tid]
                    with st.spinner(f"Analyzing {name}'s answer..."):
                        per_expert_answers[name] = answer_guide_question(q_choice2, index)
                st.session_state[cache_key] = per_expert_answers

            with st.spinner("Comparing experts..."):
                synthesis = synthesize_theme(q_choice2, per_expert_answers)

            st.markdown("### ✅ Common themes")
            themes = synthesis.get("themes", [])
            if themes:
                for t in themes:
                    experts = ", ".join(t.get("supporting_experts", []))
                    st.markdown(f"- {t.get('summary','')} *(agreed by: {experts})*")
            else:
                st.caption("No clear common theme identified.")

            st.markdown("### ⚔️ Disagreements")
            disagreements = synthesis.get("disagreements", [])
            if disagreements:
                for d in disagreements:
                    st.markdown(f"**{d.get('summary','')}**")
                    for p in d.get("positions", []):
                        st.markdown(f"- *{p.get('expert','')}*: {p.get('position','')}")
            else:
                st.caption("No clear disagreement identified.")

    # ---------------- Tab 3: Ask a Question ----------------
    with tab3:
        st.subheader("Ask a question across all 3 transcripts")
        user_q = st.text_input("Your question", placeholder="e.g. Which market has the fastest purchase timeline?")
        if st.button("Ask", key="chat_btn") and user_q.strip():
            with st.spinner("Searching all transcripts..."):
                ga = answer_chat_question(user_q, global_index)
            render_grounded_answer(ga)

    # ---------------- Tab 4: Browse Transcripts ----------------
    with tab4:
        st.subheader("Raw transcripts (for manual spot-checking)")
        tid_choice = st.selectbox("Transcript", list(turns_by_transcript.keys()))
        for t in turns_by_transcript[tid_choice]:
            speaker_label = "🎙️ Interviewer" if t.is_interviewer else f"🧑‍⚕️ {t.expert_name}"
            st.markdown(f"**[{t.timestamp}] {speaker_label}:** {t.text}")


if __name__ == "__main__":
    main()
