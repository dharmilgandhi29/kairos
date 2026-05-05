import re
import json
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Classification:
    mode: str          # "system1" or "system2"
    confidence: float  # 0.0 to 1.0
    reasoning: str     # why this classification
    requires_citation: bool
    source: str        # "heuristic" or "llm"


# --- Layer 1: Heuristic signals ---

SYSTEM1_PATTERNS = [
    r"what is \d+[\s]*[+\-*/][\s]*\d+",  # math
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
    r"\b(recession|inflation|economy|market|financial)\b.{0,30}\b(raises|lowers|crisis|crash)\b",
    r"\bhow (will|would|could|should) .+ (affect|impact|change|influence)\b",
    r"\bwhat (are|is) the (effect|impact|consequence|result) of\b",
]

UNCERTAINTY_WORDS = [
    "maybe", "might", "possibly", "unclear", "complex",
    "difficult", "challenging", "uncertain", "ambiguous"
]

DECISION_WORDS = [
    "should", "recommend", "suggest", "advise",
    "better", "best", "optimal", "choose", "decide"
]


def _heuristic_classify(task: str) -> Classification | None:
    """
    Fast rule-based classification. Returns None if uncertain.
    No API call. Instant.
    """
    task_lower = task.lower().strip()
    words = task_lower.split()

    # Check hard System 1 patterns
    for pattern in SYSTEM1_PATTERNS:
        if re.search(pattern, task_lower):
            return Classification(
                mode="system1",
                confidence=0.95,
                reasoning="Matches known simple task pattern",
                requires_citation=False,
                source="heuristic"
            )

    # Check hard System 2 patterns
    for pattern in SYSTEM2_PATTERNS:
        if re.search(pattern, task_lower):
            return Classification(
                mode="system2",
                confidence=0.90,
                reasoning="Matches known complex task pattern",
                requires_citation=True,
                source="heuristic"
            )

    # Heuristic signals
    score = 0

    # Length signal — longer tasks tend to be more complex
    if len(words) > 20:
        score += 2
    elif len(words) > 10:
        score += 1

    # Uncertainty words
    uncertainty_count = sum(1 for w in UNCERTAINTY_WORDS if w in task_lower)
    score += uncertainty_count * 2

    # Decision words
    decision_count = sum(1 for w in DECISION_WORDS if w in task_lower)
    score += decision_count * 2

    # Multiple questions
    question_count = task.count("?")
    if question_count > 1:
        score += 2

    # Sub-clauses suggest complexity
    clause_markers = ["because", "however", "although", "therefore", "given that"]
    clause_count = sum(1 for c in clause_markers if c in task_lower)
    score += clause_count

    # Clear signal
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

    # Uncertain — let LLM decide
    return None


# --- Layer 2: LLM classifier (fallback only) ---

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
    """LLM fallback for ambiguous tasks."""
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


# --- Public API ---

def classify(task: str) -> Classification:
    """
    Hybrid classifier. Heuristics first, LLM only if needed.
    """
    result = _heuristic_classify(task)

    if result is not None:
        return result

    return _llm_classify(task)