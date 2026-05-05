from dataclasses import dataclass
from kairos.core.classifier import classify
from kairos.core import system1, system2
from kairos.core.checkpoint import human_checkpoint, CheckpointResult
from kairos.core.memory import episodic
from kairos.core.memory.episodic import Episode
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
    Stores every experience in episodic memory.
    """
    # Step 1 — Classify
    classification = classify(task)
    print(f"\n[Kairos] Task classified as {classification.mode.upper()} "
          f"(confidence: {classification.confidence}, source: {classification.source})")

    # Step 2 — Execute
    checkpoint = None
    final_mode = classification.mode

    if classification.mode == "system1":
        exec_result = system1.run(task, context)

        if exec_result.confidence < human_threshold:
            print(f"[Kairos] System 1 confidence too low ({exec_result.confidence}). "
                  f"Escalating to System 2...")
            exec_result = system2.run(task, context, human_threshold)
            final_mode = "system2_escalated"
        else:
            final_mode = "system1"
    else:
        exec_result = system2.run(task, context, human_threshold)
        final_mode = "system2"

    # Step 3 — Checkpoint if needed
    requires_review = getattr(exec_result, 'requires_human_review', False)
    if auto_checkpoint and requires_review:
        checkpoint = human_checkpoint(
            task=task,
            output=exec_result.output,
            confidence=exec_result.confidence,
            reasoning_chain=getattr(exec_result, 'reasoning_chain', []),
            mode=final_mode
        )

    # Step 4 — Store in episodic memory
    episodic.initialize()

    if checkpoint:
        if checkpoint.approved and checkpoint.human_edited:
            decision = "edited"
        elif checkpoint.approved:
            decision = "approved"
        else:
            decision = "rejected"
        was_correct = checkpoint.approved and not checkpoint.human_edited
        feedback_text = checkpoint.feedback
    else:
        decision = "none"
        was_correct = True
        feedback_text = ""

    episodic.store(Episode(
        task=task,
        classification_mode=classification.mode,
        classification_source=classification.source,
        classification_confidence=classification.confidence,
        execution_mode=final_mode,
        output_confidence=exec_result.confidence,
        human_reviewed=checkpoint is not None,
        human_decision=decision,
        human_feedback=feedback_text,
        was_correct=was_correct
    ))

    # Step 5 — Give feedback to procedural memory if a pattern fired
    if getattr(classification, 'pattern_id', None):
        from kairos.core.memory.procedural import feedback as pattern_feedback
        pattern_feedback(
            pattern_id=classification.pattern_id,
            was_correct=was_correct,
            reason=feedback_text
        )

    # Step 6 — Auto-consolidate if enough episodes accumulated
    from kairos.core.memory.semantic import should_consolidate, consolidate
    from kairos.core.memory.semantic import initialize as init_semantic
    init_semantic()
    if should_consolidate():
        consolidate()

    # Step 7 — Return
    return KairosResult(
        output=checkpoint.final_output if checkpoint else exec_result.output,
        mode=final_mode,
        confidence=exec_result.confidence,
        requires_human_review=requires_review,
        reasoning_chain=getattr(exec_result, 'reasoning_chain', []),
        classification_source=classification.source,
        checkpoint_result=checkpoint
    )