from kairos import route

tasks = [
    ("What is the capital of France?", ""),
    ("Should we expand into the European market given rising inflation?", ""),
    ("What is 15 * 24?", ""),
    ("Analyze the risks of using AI in medical diagnosis", ""),
]

for task, context in tasks:
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    result = route(task, context)
    print(f"Mode: {result.mode}")
    print(f"Confidence: {result.confidence}")
    print(f"Human Review Needed: {result.requires_human_review}")
    print(f"Output: {result.output[:200]}")
    if result.reasoning_chain:
        print(f"Reasoning steps: {len(result.reasoning_chain)}")
        