import re
import json
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Classification:
    mode: str
    confidence: float
    reasoning: str
    requires_citation: bool
    source: str
    pattern_id: int = None


SYSTEM1_PATTERNS = [
    r"what is \d+[\s]*[+\-*/][\s]*\d+",
    r"what (is|are) the (capital|population|currency) of",
    r"how (many|much) (days|hours|minutes|years)",
    r"what (day|time|date) is",
    r"(translate|convert) .+ to .+",
    r"define (the word|the term)?",
    r"what does .+ stand for",
    r"spell .+",
]

SYSTEM2_PATTERNS = [
    r"\b(should|would you recommend|is it better)\b",
    r"\b(analyze|analyse|evaluation|evaluate)\b",
    r"\b(ethical|moral|implications|consequences)\b",
    r"\b(strategy|strategic|plan|roadmap)\b",
    r"\b(compare and contrast|pros and cons|tradeoffs)\b",
    r"\b(why did|why does|why would)\b",
    r"\b(predict|forecast|future of)\b",
    r"\b(risk|danger|impact of)\b",
    r"\bwhat will happen (to|if|when)\b",
    r"\bwhat (would|could) happen\b",
    r"\bif .+ (raises|lowers|increases|decreases|changes)\b",
    r"\b(effect|effects|impact|impacts) of\b",
    r"\bhow (will|would|could|should) .+ (affect|impact|change|influence)\b",
    r"\bwhat (are|is) the (effect|impact|consequence|result) of\b",
    r"\b(acquiring|acquisition|invest|investment|valuation)\b",
    r"\b(wise|prudent|advisable|recommended)\b.{0,20}\b(decision|choice|move)\b",
]

UNCERTAINTY_WORDS = [
    "maybe", "might", "possibly", "unclear", "complex",
    "difficult", "challenging", "uncertain", "ambiguous"
]

DECISION_WORDS = [
    "should", "recommend", "suggest", "advise",
    "better", "best", "optimal", "choose", "decide"
]


def _heuristic_classify(task: str):
    task_lower = task.lower().strip()
    words = task_lower.split()

    for pattern in SYSTEM1_PATTERNS:
        if re.search(pattern, task_lower):
            return Classification(
                mode="system1",
                confidence=0.95,
                reasoning="Matches known simple task pattern",
                requires_citation=False,
                source="heuristic"
            )

    for pattern in SYSTEM2_PATTERNS:
        if re.search(pattern, task_lower):
            return Classification(
                mode="system2",
                confidence=0.90,
                reasoning="Matches known complex task pattern",
                requires_citation=True,
                source="heuristic"
            )

    score = 0
    if len(words) > 20:
        score += 2
    elif len(words) > 10:
        score += 1

    score += sum(1 for w in UNCERTAINTY_WORDS if w in task_lower) * 2
    score += sum(1 for w in DECISION_WORDS if w in task_lower) * 2

    if task.count("?") > 1:
        score += 2

    clause_markers = ["because", "however", "although", "therefore", "given that"]
    score += sum(1 for c in clause_markers if c in task_lower)

    if score >= 4:
        return Classification(
            mode="system2",
            confidence=0.75,
            reasoning=f"Heuristic signals: complexity score {score}",
            requires_citation=True,
            source="heuristic"
        )
    elif score <= 1:
        return Classification(
            mode="system1",
            confidence=0.75,
            reasoning=f"Heuristic signals: low complexity score {score}",
            requires_citation=False,
            source="heuristic"
        )

    return None


CLASSIFIER_PROMPT = """You are a task complexity classifier.

Analyze the task and return ONLY this JSON:
{
    "mode": "system1" or "system2",
    "confidence": 0.0 to 1.0,
    "reasoning": "one sentence explanation",
    "requires_citation": true or false
}

system1 = fast, simple, routine, single-step, factual
system2 = complex, multi-step, ambiguous, high-stakes, requires judgment

Return ONLY valid JSON. Nothing else."""


def _llm_classify(task: str) -> Classification:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        SystemMessage(content=CLASSIFIER_PROMPT),
        HumanMessage(content=task)
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)
    return Classification(
        mode=data["mode"],
        confidence=data["confidence"],
        reasoning=data["reasoning"],
        requires_citation=data["requires_citation"],
        source="llm"
    )


def classify(task: str) -> Classification:
    """
    Hybrid classifier. Three layers:
    0. Procedural memory (learned patterns)
    1. Heuristics (hardcoded rules)
    2. LLM fallback
    """
    # Layer 0 — Procedural memory
    try:
        from kairos.core.memory.procedural import apply_to_classifier
        from kairos.core.memory.semantic import initialize as init_semantic
        init_semantic()
        learned, pattern_id = apply_to_classifier(task)
        if learned:
            return Classification(
                mode=learned,
                confidence=0.85,
                reasoning="Matched learned pattern from past experience",
                requires_citation=learned == "system2",
                source="procedural_memory",
                pattern_id=pattern_id
            )
    except Exception:
        pass

    # Layer 1 — Heuristics
    result = _heuristic_classify(task)
    if result is not None:
        return result

    # Layer 2 — LLM fallback
    return _llm_classify(task)