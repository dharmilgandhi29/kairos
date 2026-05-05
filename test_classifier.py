from kairos.core.classifier import classify

tasks = [
    "What is 2 + 2?",
    "What is the capital of France?",
    "Should our company acquire this startup?",
    "Analyze the ethical implications of AI in criminal sentencing",
    "Send an email to John saying hello",
    "What will happen to inflation if the Fed raises rates by 2% during a recession?",
    "Translate hello to Spanish",
    "Is it better to use PostgreSQL or MongoDB for our new app given we have complex queries?",
]

print(f"{'Task':<55} {'Mode':<10} {'Conf':<6} {'Source':<12} Reasoning")
print("-" * 110)

for task in tasks:
    result = classify(task)
    print(f"{task[:54]:<55} {result.mode:<10} {result.confidence:<6} {result.source:<12} {result.reasoning}")