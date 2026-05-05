from dataclasses import dataclass
from kairos.core.classifier import classify
from kairos.core import system1, system2
from dotenv import load_dotenv

load_dotenv()

@dataclass
class KairosResult:
    output: str
    mode: str
    confidence: float
    requires_human_review: bool
    reasoning_chain: list
    classification_source: str


def route(task: str, context: str = "", human_threshold: float = 0.6) -> KairosResult:
    """
    Main router. Classifies task then executes
    with the appropriate reasoning depth.
    """
    # Step 1 — Classify
    classification = classify(task)

    print(f"\n[Kairos] Task classified as {classification.mode.upper()} "
          f"(confidence: {classification.confidence}, source: {classification.source})")

    # Step 2 — Execute
    if classification.mode == "system1":
        result = system1.run(task, context)

        # If System 1 confidence is low, escalate to System 2
        if result.confidence < human_threshold:
            print(f"[Kairos] System 1 confidence too low ({result.confidence}). "
                  f"Escalating to System 2...")
            result2 = system2.run(task, context, human_threshold)
            return KairosResult(
                output=result2.output,
                mode="system2_escalated",
                confidence=result2.confidence,
                requires_human_review=result2.requires_human_review,
                reasoning_chain=result2.reasoning_chain,
                classification_source=classification.source
            )

        return KairosResult(
            output=result.output,
            mode="system1",
            confidence=result.confidence,
            requires_human_review=False,
            reasoning_chain=[],
            classification_source=classification.source
        )

    else:
        result = system2.run(task, context, human_threshold)
        return KairosResult(
            output=result.output,
            mode="system2",
            confidence=result.confidence,
            requires_human_review=result.requires_human_review,
            reasoning_chain=result.reasoning_chain,
            classification_source=classification.source
        )