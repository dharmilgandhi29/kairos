from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

@dataclass
class System2Result:
    output: str
    confidence: float
    reasoning_chain: list
    citations: list
    requires_human_review: bool
    mode: str = "system2"


def run(task: str, context: str = "", human_threshold: float = 0.6) -> System2Result:
    """
    System 2 — Slow, deliberate, thorough reasoning.
    Used for complex, ambiguous, high-stakes tasks.
    Citations required. Step by step reasoning.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    system_prompt = """You are a careful, deliberate AI analyst.

For every task you must:
1. Break the problem into steps
2. Reason through each step explicitly
3. Cite your reasoning — if you make a claim, explain why
4. Acknowledge uncertainty honestly
5. Never guess — if you don't know something, say so

Format your response as:

REASONING:
[step by step thinking]

ANSWER:
[your final answer]

CONFIDENCE: [0.0 to 1.0]
UNCERTAINTIES: [what you are not sure about, or "none"]"""

    user_message = task
    if context:
        user_message = f"Context: {context}\n\nTask: {task}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # Parse the structured response
    reasoning_chain = []
    answer = raw
    confidence = 0.7
    uncertainties = []

    if "REASONING:" in raw:
        parts = raw.split("ANSWER:")
        reasoning_text = parts[0].replace("REASONING:", "").strip()
        reasoning_chain = [line.strip() for line in reasoning_text.split("\n") if line.strip()]

        if len(parts) > 1:
            answer_part = parts[1]

            if "CONFIDENCE:" in answer_part:
                answer_conf = answer_part.split("CONFIDENCE:")
                answer = answer_conf[0].strip()
                conf_text = answer_conf[1].split("UNCERTAINTIES:")[0].strip()
                try:
                    confidence = float(conf_text)
                except:
                    confidence = 0.7

            if "UNCERTAINTIES:" in answer_part:
                uncertainty_text = answer_part.split("UNCERTAINTIES:")[1].strip()
                if uncertainty_text.lower() != "none":
                    uncertainties = [uncertainty_text]

    return System2Result(
        output=answer,
        confidence=confidence,
        reasoning_chain=reasoning_chain,
        citations=uncertainties,
        requires_human_review=confidence < human_threshold
    )
    