from kairos import route

# This should trigger human review — complex task, confidence will be below threshold
task = "Should we shut down our worst performing product line given current market conditions?"

result = route(task, human_threshold=0.75)

print(f"\nFinal Output: {result.output[:300]}")
print(f"Mode: {result.mode}")
print(f"Human reviewed: {result.checkpoint_result is not None}")