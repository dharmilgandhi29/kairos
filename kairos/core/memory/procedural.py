def get_learned_heuristics() -> dict:
    from kairos.core.memory.semantic import get_patterns
    patterns = get_patterns()
    system1_keywords = []
    system2_keywords = []

    for p in patterns:
        if p["confidence"] >= 0.75:
            words = p["pattern"].lower().split()
            keywords = [w for w in words if len(w) > 4]
            if p["classification"] == "system1":
                system1_keywords.extend(keywords)
            else:
                system2_keywords.extend(keywords)

    return {
        "system1": list(set(system1_keywords)),
        "system2": list(set(system2_keywords))
    }


def apply_to_classifier(task: str) -> tuple:
    from kairos.core.memory.semantic import get_patterns
    patterns = get_patterns()
    task_lower = task.lower()
    task_words = set(task_lower.split())

    best_match = None
    best_confidence = 0
    best_pattern_id = None

    for p in patterns:
        if p["confidence"] <= best_confidence:
            continue

        pattern_words = [w for w in p["pattern"].lower().split() if len(w) > 4]
        pattern_overlap = sum(1 for w in pattern_words if w in task_lower)
        if pattern_overlap >= 2:
            best_match = p["classification"]
            best_confidence = p["confidence"]
            best_pattern_id = p["id"]
            continue

        for example in p.get("examples", []):
            example_words = set(example.lower().split())
            overlap = len(example_words & task_words)
            significant_overlap = any(
                w in task_lower for w in example_words
                if len(w) > 5
            )
            if overlap >= 2 or significant_overlap:
                if p["confidence"] > best_confidence:
                    best_match = p["classification"]
                    best_confidence = p["confidence"]
                    best_pattern_id = p["id"]
                    break

    if best_confidence >= 0.7:
        return best_match, best_pattern_id
    return None, None


def feedback(pattern_id: int, was_correct: bool, reason: str = ""):
    from kairos.core.memory.semantic import decay_pattern, reinforce_pattern
    if was_correct:
        reinforce_pattern(pattern_id)
    else:
        decay_pattern(pattern_id, reason)