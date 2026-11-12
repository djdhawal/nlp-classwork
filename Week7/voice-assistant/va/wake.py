"""Wake word matching on speech-to-text output.

Accepts:  "Loop, book ...", "Okay, loop, book ...", "... tomorrow at 3. Loop. Book ..."
Ignores:  "I like to loop videos"   (the wake word is in the middle of a clause)
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[A-Za-z0-9']+|[.!?;:,]")
_SENTENCE_END = set(".!?;:")


class WakeWord:
    def __init__(self, cfg: dict):
        w = cfg["wake"]
        self.variants = {v.lower() for v in w["variants"]} | {w["word"].lower()}
        self.fillers = {f.lower() for f in w.get("fillers", [])}

    def find(self, text: str) -> str | None:
        """Return the command after the wake word ("" if only the wake word), or None if there is no wake word."""
        spans = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text)]
        for i, (tok, _, end) in enumerate(spans):
            if tok.lower() not in self.variants:
                continue
            # Words since the start of this sentence, before the wake word.
            j, before = i - 1, []
            while j >= 0 and spans[j][0] not in _SENTENCE_END:
                if spans[j][0] != ",":
                    before.append(spans[j][0].lower())
                j -= 1
            followed_by_punct = i + 1 < len(spans) and spans[i + 1][0] in _SENTENCE_END | {","}
            prev_is_comma = i > 0 and spans[i - 1][0] == ","
            at_start = all(b in self.fillers for b in before)        # "Loop ...", "Okay, loop ..."
            vocative = prev_is_comma and followed_by_punct          # "..., loop, book ..."
            if at_start or vocative:
                return text[end:].lstrip(" ,.!?;:-").strip()
        return None
