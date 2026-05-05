from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

@dataclass
class System1Result:
    output: str
    confidence: float
    mode: str = "system1"
    citations: list = None


def run(task: str, context: str = "") -> System1Result:
    """
    System 1 — Fast, cheap, direct response.
    Used for simple, routine, well-defined tasks.
    No citation required. Minimal tokens.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system_prompt = """You are a fast, efficient AI assistant.
Answer directly and concisely. No unnecessary explanation.
If you are not confident, say so clearly instead of guessing."""

    user_message = task
    if context:
        user_message = f"Context: {context}\n\nTask: {task}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]

    response = llm.invoke(messages)

    # Simple confidence heuristic
    output = response.content.strip()
    confidence = 0.5
    uncertain_phrases = [
        "i'm not sure", "i don't know", "unclear",
        "it depends", "i cannot", "i'm unable"
    ]
    if any(p in output.lower() for p in uncertain_phrases):
        confidence = 0.4
    else:
        confidence = 0.85

    return System1Result(
        output=output,
        confidence=confidence
    )