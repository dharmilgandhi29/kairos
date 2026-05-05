from dataclasses import dataclass
from kairos.core.classifier import classify
from kairos.core import system1, system2
from kairos.core.checkpoint import human_checkpoint, CheckpointResult
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
    checkpoint_result: CheckpointResult = None


def route(
    task: str,
    context: str = "",
    human_threshold: float = 0.6,
    auto_checkpoint: bool = True
) -> KairosResult:
    """
    Main router. Classifies → executes → checkpoints if needed.
    """
    # Step 1 — Classify
    classification = classify(task)

    print(f"\n[Kairos] Task classified as {classification.mode.upper()} "
          f"(confidence: {classification.confidence}, source: {classification.source})")

    # Step 2 — Execute
    if classification.mode == "system1":
        result = system1.run(task, context)

        if result.confidence < human_threshold:
            print(f"[Kairos] System 1 confidence too low ({result.confidence}). "
                  f"Escalating to System 2...")
            s2_result = system2.run(task, context, human_threshold)

            checkpoint = None
            if auto_checkpoint and s2_result.requires_human_review:
                checkpoint = human_checkpoint(
                    task=task,
                    output=s2_result.output,
                    confidence=s2_result.confidence,
                    reasoning_chain=s2_result.reasoning_chain,
                    mode="system2_escalated"
                )

            return KairosResult(
                output=checkpoint.final_output if checkpoint else s2_result.output,
                mode="system2_escalated",
                confidence=s2_result.confidence,
                requires_human_review=s2_result.requires_human_review,
                reasoning_chain=s2_result.reasoning_chain,
                classification_source=classification.source,
                checkpoint_result=checkpoint
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

        checkpoint = None
        if auto_checkpoint and result.requires_human_review:
            checkpoint = human_checkpoint(
                task=task,
                output=result.output,
                confidence=result.confidence,
                reasoning_chain=result.reasoning_chain,
                mode="system2"
            )

        return KairosResult(
            output=checkpoint.final_output if checkpoint else result.output,
            mode="system2",
            confidence=result.confidence,
            requires_human_review=result.requires_human_review,
            reasoning_chain=result.reasoning_chain,
            classification_source=classification.source,
            checkpoint_result=checkpoint
        )