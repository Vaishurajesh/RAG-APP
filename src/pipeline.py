"""
pipeline.py
-----------
Three capabilities, all built on the same grounding pattern:

    retrieve turns -> ask LLM for {answer, quotes[{ref, quote}]}
    -> validate each quote is an exact substring of the turn it cites
    -> attach timestamp/speaker from the Turn object (never from the LLM)

1. answer_guide_question()   - one expert, one interview-guide question
2. synthesize_theme()        - compare 3 experts' grounded answers on one question
3. answer_chat_question()    - free-form Q&A across all transcripts
"""
import re
from dataclasses import dataclass
from typing import List, Dict

from .parser import Turn
from .retrieval import TurnIndex
from .llm_client import call_json

GUIDE_QA_SYSTEM = """You are a careful research analyst reviewing ONE expert interview transcript excerpt.
You will be given a research question and a numbered list of transcript excerpts from a single expert.
Answer using ONLY the information in the excerpts. Do not use outside knowledge and do not invent facts, numbers or quotes.

Return STRICT JSON only, no markdown, matching this schema:
{
  "answer": "2-4 sentence answer in your own words, grounded only in the excerpts",
  "quotes": [{"ref": <int>, "quote": "<verbatim substring copied EXACTLY from that excerpt's text>"}],
  "insufficient_evidence": <true or false>
}
Rules:
- "ref" must be the excerpt number shown in brackets, e.g. [3].
- Each "quote" must be an exact, contiguous, character-for-character substring of that excerpt's text. Never paraphrase inside a quote. Never combine text from two excerpts into one quote.
- Return 1-3 of the strongest supporting quotes. Prefer the expert's own words, not the interviewer's question.
- If the excerpts do not address the question, set "insufficient_evidence": true, "answer" explaining that, and "quotes": []."""

CHAT_SYSTEM = """You are a careful research analyst answering a question using excerpts from THREE expert interview transcripts (different European markets).
Answer using ONLY the information in the excerpts below. Do not use outside knowledge and do not invent facts, numbers, or quotes.
If the experts disagree, say so explicitly and attribute each position to the correct expert.

Return STRICT JSON only, no markdown, matching this schema:
{
  "answer": "A clear, direct answer in your own words, grounded only in the excerpts. Note any disagreement between experts.",
  "quotes": [{"ref": <int>, "quote": "<verbatim substring copied EXACTLY from that excerpt's text>"}],
  "insufficient_evidence": <true or false>
}
Rules:
- "ref" is the excerpt number shown in brackets, e.g. [3].
- Each "quote" must be an exact, contiguous substring of that excerpt's text.
- Return up to 4 supporting quotes, ideally covering more than one expert if relevant.
- If nothing in the excerpts answers the question, set "insufficient_evidence": true."""

SYNTHESIS_SYSTEM = """You are comparing how three experts (different markets) answered the SAME research question.
You are given each expert's grounded answer plus their exact supporting quotes. Base your comparison ONLY on this material -- do not invent new facts.

Return STRICT JSON only, no markdown:
{
  "themes": [{"summary": "<a point of agreement across 2 or more experts, in your own words>", "supporting_experts": ["<expert name>", "..."]}],
  "disagreements": [{"summary": "<what they disagree about, in your own words>", "positions": [{"expert": "<name>", "position": "<their position, in your own words>"}]}]
}
Rules:
- Only name experts who are actually present in the input.
- If there is genuinely no disagreement, return an empty "disagreements" list. Do not manufacture one.
- Keep each summary to one sentence."""


@dataclass
class Citation:
    market: str
    expert_name: str
    timestamp: str
    quote: str
    verified: bool


@dataclass
class GroundedAnswer:
    answer: str
    citations: List[Citation]
    insufficient_evidence: bool


def _build_context(turns: List[Turn]) -> str:
    lines = []
    for i, t in enumerate(turns):
        who = "Interviewer" if t.is_interviewer else t.expert_name
        lines.append(f"[{i}] ({t.timestamp}) {who}: {t.text}")
    return "\n".join(lines)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _validate_quotes(raw_quotes: List[dict], turns: List[Turn]) -> List[Citation]:
    """Exact-substring check. This is what prevents fabricated quotes from
    ever reaching the UI -- if the model's 'quote' isn't literally in the
    turn text it claims to cite, we drop it rather than trust it."""
    citations = []
    for q in raw_quotes:
        ref = q.get("ref")
        quote = (q.get("quote") or "").strip()
        if ref is None or not quote or ref < 0 or ref >= len(turns):
            continue
        turn = turns[ref]
        verified = _normalize(quote) in _normalize(turn.text)
        if verified:
            citations.append(
                Citation(
                    market=turn.market,
                    expert_name=turn.expert_name,
                    timestamp=turn.timestamp,
                    quote=quote,
                    verified=True,
                )
            )
        # unverified quotes are silently dropped -- never shown to the user
    return citations


def _run_grounded(system_prompt: str, user_header: str, turns: List[Turn]) -> GroundedAnswer:
    context = _build_context(turns)
    user_prompt = f"{user_header}\n\nExcerpts:\n{context}"
    result = call_json(system_prompt, user_prompt)
    citations = _validate_quotes(result.get("quotes", []), turns)
    return GroundedAnswer(
        answer=result.get("answer", "").strip(),
        citations=citations,
        insufficient_evidence=bool(result.get("insufficient_evidence", False)),
    )


def answer_guide_question(question: str, index: TurnIndex, k: int = 6) -> GroundedAnswer:
    turns = index.search(question, k=k)
    header = f"Research question: {question}"
    return _run_grounded(GUIDE_QA_SYSTEM, header, turns)


def answer_chat_question(question: str, index: TurnIndex, k: int = 10) -> GroundedAnswer:
    turns = index.search(question, k=k)
    header = f"Question: {question}"
    return _run_grounded(CHAT_SYSTEM, header, turns)


def synthesize_theme(question: str, per_expert_answers: Dict[str, GroundedAnswer]) -> dict:
    """per_expert_answers: {expert_name: GroundedAnswer} for one guide question."""
    blocks = [f"Research question: {question}\n"]
    for expert, ga in per_expert_answers.items():
        blocks.append(f"--- {expert} ---")
        blocks.append(f"Grounded answer: {ga.answer}")
        for c in ga.citations:
            blocks.append(f'  Quote ({c.timestamp}): "{c.quote}"')
    user_prompt = "\n".join(blocks)
    return call_json(SYNTHESIS_SYSTEM, user_prompt)
