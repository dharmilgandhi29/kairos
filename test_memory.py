from kairos.core.memory import episodic
import json

# Initialize
episodic.initialize()

# Run a few tasks through the router first to generate episodes
from kairos import route

tasks = [
    "What is the capital of France?",
    "Should we acquire this startup given market conditions?",
    "What is 15 * 24?",
]

for task in tasks:
    print(f"\nRunning: {task}")
    route(task, auto_checkpoint=False)

# Check what got stored
print("\n--- Episodic Memory Stats ---")
stats = episodic.get_stats()
for key, val in stats.items():
    print(f"  {key}: {val}")

print("\n--- Similar to 'market' ---")
similar = episodic.get_similar("market conditions strategy")
for ep in similar:
    print(f"  Task: {ep.task[:60]}")
    print(f"  Mode: {ep.execution_mode}, Correct: {ep.was_correct}")

    from kairos.core.memory import semantic

print("\n--- Running Consolidation ---")
semantic.initialize()
result = semantic.consolidate()
print(json.dumps(result, indent=2))

print("\n--- Learned Patterns ---")
patterns = semantic.get_patterns()
for p in patterns:
    print(f"  Pattern: {p['pattern'][:60]}")
    print(f"  Class: {p['classification']}, Confidence: {p['confidence']}")


from kairos.core.classifier import classify

print("\n--- Testing Procedural Memory in Classifier ---")
test_tasks = [
    "What is 99 * 12?",           # should match math pattern
    "What is the capital of Japan?", # should match geography pattern
    "Should we invest in this company given market trends?",  # should match business pattern
]

for task in test_tasks:
    result = classify(task)
    print(f"\nTask: {task}")
    print(f"Mode: {result.mode}, Source: {result.source}, Confidence: {result.confidence}")



print("\n--- Testing Procedural Memory with Novel Tasks ---")
novel_tasks = [
    "Calculate 847 divided by 23",          # math but different phrasing
    "Name the capital city of Argentina",    # geography but different phrasing  
    "Is acquiring TechCorp a wise decision given current valuations?",  # business different phrasing
]

for task in novel_tasks:
    result = classify(task)
    print(f"\nTask: {task}")
    print(f"Mode: {result.mode}, Source: {result.source}, Confidence: {result.confidence}")