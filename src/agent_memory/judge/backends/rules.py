"""Deterministic baseline: answers from the trigger flags and tool facts alone. The floor."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..lesson import cited_entries
from ..questions import Question
from .base import Answer, JudgeInput, answer_from_probabilities

_ACTION_WORDS = re.compile(
    r"\b(use|run|check|avoid|prefer|always|never|before|after|verify|confirm|read|install|"
    r"do not|don't|instead)\b",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]{2,}")


class RulesBackend:
    name = "rules"
    local = True

    def decide(self, inp: JudgeInput, questions: Sequence[Question]) -> dict[str, Answer]:
        flags = set(inp.flags)
        facts = inp.facts
        out: dict[str, Answer] = {}
        for q in questions:
            probs = self._probabilities(q, flags, facts, inp)
            out[q.id] = answer_from_probabilities(q, probs, method="rules")
        return out

    def _probabilities(
        self, q: Question, flags: set[str], facts: dict, inp: JudgeInput
    ) -> dict[str, float]:
        hard_fail = bool(flags & {"T1", "T2", "T4"})
        if q.id == "failed":
            p = 0.9 if hard_fail else 0.6 if "T3" in flags else 0.1
            return {"yes": p, "no": 1 - p}
        if q.id == "incomplete":
            cut_off = facts.get("closed_by") in ("next_message", "session_end")
            p = 0.8 if cut_off else 0.1
            return {"yes": p, "no": 1 - p}
        if q.id == "user_corrected":
            p = 0.9 if "T4" in flags else 0.1
            return {"yes": p, "no": 1 - p}
        if q.id == "env_confirmed":
            p = 0.7 if facts.get("last_result_ok") and not hard_fail else 0.3
            return {"yes": p, "no": 1 - p}
        if q.id == "category":
            if "T2" in flags:
                pick = "environment_or_provider"
            elif "T3" in flags:
                pick = "tool_misuse"
            elif "T4" in flags:
                pick = "wrong_assumption"
            elif "T1" in flags:
                pick = "user_redirect"
            else:
                pick = "none"
            rest = (1 - 0.7) / (len(q.options()) - 1)
            return {o: (0.7 if o == pick else rest) for o in q.options()}
        if q.id == "recurrence":
            p = 0.8 if facts.get("repeated_tool_error") else 0.3
            return {"yes": p, "no": 1 - p}
        if q.id.startswith(("grounded.", "faithful.")):
            claim = inp.claims[int(q.id.rsplit(".", 1)[1]) - 1]
            evidence = "\n".join(body for _, body in cited_entries(inp.state, claim.cited))
            supported = self._overlap(claim.text, evidence, facts)
            if not claim.cited:
                p = 0.1
            elif q.id.startswith("grounded."):
                p = 0.8 if supported else 0.2
            else:
                p = 0.65 if supported else 0.35
            return {"yes": p, "no": 1 - p}
        supported = self._draft_mentions_turn(inp)
        if q.id in ("supported", "grounded"):
            p = 0.8 if supported else 0.2
            return {"yes": p, "no": 1 - p}
        if q.id == "faithful":
            # Nothing in the facts can check a reading of the evidence; the overlap signal is
            # reused at lower confidence so the floor stays a floor.
            p = 0.65 if supported else 0.35
            return {"yes": p, "no": 1 - p}
        if q.id == "rule_supported":
            # Token overlap cannot establish an inference from true claims to an exact rule.
            return {"yes": 0.5, "no": 0.5}
        if q.id == "generalizable":
            return {"0": 0.3, "1": 0.6, "2": 0.1} if supported else {"0": 0.7, "1": 0.2, "2": 0.1}
        if q.id == "actionable":
            p = 0.7 if inp.draft and _ACTION_WORDS.search(inp.draft) else 0.3
            return {"yes": p, "no": 1 - p}
        if q.id == "enforced":
            agent_started = str(facts.get("caller") or "human").startswith("agent")
            pick = "harness" if agent_started else "none"
            rest = (1 - 0.6) / (len(q.options()) - 1)
            return {o: (0.6 if o == pick else rest) for o in q.options()}
        return dict.fromkeys(q.options(), 1.0)

    @staticmethod
    def _draft_mentions_turn(inp: JudgeInput) -> bool:
        if not inp.draft:
            return False
        state_tokens = {t.lower() for t in _TOKEN.findall(inp.state)}
        draft_tokens = {t.lower() for t in _TOKEN.findall(inp.draft)}
        names = {str(t).lower() for t in inp.facts.get("tools", [])}
        return bool((draft_tokens & names) or len(draft_tokens & state_tokens) >= 3)

    @staticmethod
    def _overlap(text: str, evidence: str, facts: dict) -> bool:
        state_tokens = {t.lower().rstrip(".") for t in _TOKEN.findall(evidence)}
        draft_tokens = {t.lower().rstrip(".") for t in _TOKEN.findall(text)}
        names = {str(t).lower() for t in facts.get("tools", [])} & state_tokens
        return bool((draft_tokens & names) or len(draft_tokens & state_tokens) >= 3)
