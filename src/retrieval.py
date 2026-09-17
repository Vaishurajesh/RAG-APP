"""
retrieval.py
------------
Lightweight retrieval over transcript turns.

Why TF-IDF instead of embeddings?
For a case study with 3 transcripts (~14 turns each, ~42 turns total),
an embedding API call per query is unnecessary latency/cost/complexity.
TF-IDF + cosine similarity is deterministic, needs no network call, and
is more than accurate enough at this corpus size. The retrieval layer is
isolated behind a simple function so it can be swapped for a vector DB
(e.g. pgvector, FAISS + embeddings) without touching the rest of the
pipeline -- that swap is exactly what I'd do first when scaling to 30+
transcripts (see README "Scaling" section).
"""
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .parser import Turn


class TurnIndex:
    """
    Retrieval unit = the EXPERT's answer turn, but the text used for TF-IDF
    matching is "preceding interviewer question + the answer itself".

    Why: interview transcripts strictly alternate Q/A. A query like "what
    are the main barriers?" often lexically matches the *interviewer's*
    question turn ("What are the main barriers?") better than it matches
    the expert's actual answer turn -- which is the one we actually want
    to cite. Folding the question into the answer's search text fixes
    recall without ever letting the interviewer's turn become a quotable
    citation (quotes should always be the expert's own words).
    """

    def __init__(self, turns: List[Turn]):
        self.turns = [t for t in turns if not t.is_interviewer]
        # Key by (transcript_id, turn_index) so this also works for a GLOBAL
        # index spanning multiple transcripts at once.
        by_key = {(t.transcript_id, t.turn_index): t for t in turns}
        search_corpus = []
        for t in self.turns:
            preceding = by_key.get((t.transcript_id, t.turn_index - 1))
            q_text = preceding.text if (preceding and preceding.is_interviewer) else ""
            search_corpus.append(f"{q_text} {t.text}")

        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        self.matrix = self.vectorizer.fit_transform(search_corpus) if search_corpus else None

    def search(self, query: str, k: int = 6) -> List[Turn]:
        if not self.turns or self.matrix is None:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        ranked = sorted(range(len(self.turns)), key=lambda i: sims[i], reverse=True)
        top = [self.turns[i] for i in ranked[:k] if sims[i] > 0]
        return top if top else self.turns[:k]


def build_indexes(turns_by_transcript: Dict[str, List[Turn]]) -> Dict[str, TurnIndex]:
    """One index per transcript (used for per-expert guide answers)."""
    return {tid: TurnIndex(turns) for tid, turns in turns_by_transcript.items()}


def build_global_index(turns_by_transcript: Dict[str, List[Turn]]) -> TurnIndex:
    """One index across ALL transcripts (used for the free-form Q&A chat)."""
    all_turns = [t for turns in turns_by_transcript.values() for t in turns]
    return TurnIndex(all_turns)
