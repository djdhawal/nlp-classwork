"""Section 3.2 step 8: three guardrails around the RAG flow.

1. Input guard (before retrieval): empty / too long queries, prompt-injection phrases, personal data.
   Why: OWASP's Top 10 for LLM apps lists prompt injection as risk #1 and sensitive-information
   disclosure as #2. A news assistant has no reason to receive SSNs or "ignore your instructions".
2. Retrieval guard (before generation): abstain when even the best passage is a weak match, and
   remove sentences in the passages that look like injected instructions.
   Why: when retrieval returns off-topic chunks, the LLM tends to answer from memory anyway and
   attach citations that do not support the claim. Abstaining early is cheaper and more honest.
   Scraped web text is untrusted too ("indirect prompt injection"): a small model can follow an
   instruction hidden in an article even when the prompt says not to.
3. Output guard (after generation): every cited id must be one of the retrieved passages, and an
   answer that is not the refusal must carry at least one citation.
   Why: invented citations look trustworthy but cannot be checked; an uncited answer is most
   likely the model's own knowledge, which is exactly what grounding is meant to prevent.

Each guard returns a GuardResult(ok, reason, details). ok=False means "block or replace".
"""
import re
from dataclasses import dataclass, field

from . import config
from .prompts import NOT_FOUND, extract_citations


# bge-small cosine on this corpus: off-topic chunks score about 0.55-0.59, relevant ones 0.65-0.85,
# so config.MIN_RETRIEVAL_SCORE (0.55) lets off-topic questions through. 0.62 sits in the gap.
MIN_SCORE = 0.62


@dataclass
class GuardResult:
    ok: bool
    reason: str = ""
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 1. input guard
INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any |the )?(previous|prior|above|system) ",
    r"forget (everything|all|your) (you|instructions|rules)",
    r"(reveal|show|print|repeat) (me )?(your|the) (system )?(prompt|instructions)",
    r"you are now (a|an|in) ",
    r"\b(jailbreak|DAN mode|developer mode)\b",
    r"act as (if you|an? unrestricted)",
    r"</?(system|passage)>",
]
_INJECTION = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# High-risk identifiers are refused; contact details are masked and the query goes ahead.
BLOCK_PII = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}
MASK_PII = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b"),
}


def check_input(query):
    """Validate a user query. details['query'] holds the (possibly PII-masked) query to use."""
    q = (query or "").strip()
    if not q:
        return GuardResult(False, "empty query", {"query": q})
    if len(q) > config.MAX_QUERY_CHARS:
        return GuardResult(False, f"query too long ({len(q)} > {config.MAX_QUERY_CHARS} chars)", {"query": q})
    hits = [p.pattern for p in _INJECTION if p.search(q)]
    if hits:
        return GuardResult(False, "possible prompt injection", {"query": q, "patterns": hits})
    blocked = [name for name, rx in BLOCK_PII.items() if rx.search(q)]
    if blocked:
        return GuardResult(False, f"sensitive personal data in query ({', '.join(blocked)})", {"query": q})
    masked, found = q, []
    for name, rx in MASK_PII.items():
        if rx.search(masked):
            found.append(name)
            masked = rx.sub(f"[{name.upper()}]", masked)
    if found:
        return GuardResult(True, f"PII masked ({', '.join(found)})", {"query": masked, "masked": found})
    return GuardResult(True, "ok", {"query": q})


# ---------------------------------------------------------------- 2. retrieval guard
def dense_score(hit):
    return hit.get("dense_score", hit["score"])


def check_retrieval(hits, min_score=MIN_SCORE):
    """Abstain when nothing was retrieved or the best cosine score is below `min_score`.

    Uses the bi-encoder cosine: `dense_score` when present (after re-ranking `score` is a
    cross-encoder logit on a different scale), else `score`."""
    if not hits:
        return GuardResult(False, "no passages retrieved", {"best_score": None})
    best = max(dense_score(h) for h in hits)
    injected = [h["chunk_id"] for h in hits if any(p.search(h["text"]) for p in _INJECTION)]
    details = {"best_score": best, "min_score": min_score, "injected": injected}
    if best < min_score:
        return GuardResult(False, f"best retrieval score {best:.3f} < {min_score}", details)
    return GuardResult(True, "ok" if not injected else f"ok (instructions removed from {injected})", details)


_SENT = re.compile(r"(?<=[.!?])\s+")


def clean_hits(hits):
    """Copies of the hits with instruction-like sentences removed from the passage text."""
    out = []
    for h in hits:
        sents = _SENT.split(h["text"])
        kept = [s for s in sents if not any(p.search(s) for p in _INJECTION)]
        out.append(dict(h, text=" ".join(kept)) if len(kept) < len(sents) else h)
    return out


# ---------------------------------------------------------------- 3. output guard
# Small models often abstain in their own words ("There is no information about ...") and still
# add citations. An answer that *starts* like this is treated as an abstention.
SOFT_REFUSAL = re.compile(r"^\W*(there is no (information|mention)|(the )?(provided )?(sources|passages) do not "
                          r"(contain|mention|provide)|no information|i (cannot|can't|don't) (find|answer|know))",
                          re.IGNORECASE)


def is_refusal(answer):
    a = (answer or "").strip()
    return NOT_FOUND.lower().rstrip(".") in a.lower() and len(a) < len(NOT_FOUND) + 40 or bool(SOFT_REFUSAL.search(a))


def check_output(answer, hits):
    """Check the citations of a generated answer against the retrieved hits."""
    retrieved = {h["chunk_id"] for h in hits}
    cited = extract_citations(answer)
    valid = [c for c in cited if c in retrieved]
    invalid = [c for c in cited if c not in retrieved]
    details = {"cited": cited, "valid": valid, "invalid": invalid, "refusal": is_refusal(answer)}
    if details["refusal"]:
        return GuardResult(True, "model abstained", details)
    if invalid:
        return GuardResult(False, f"cites passages that were not retrieved: {invalid}", details)
    if not valid:
        return GuardResult(False, "answer has no citations (likely not grounded)", details)
    return GuardResult(True, "ok", details)
