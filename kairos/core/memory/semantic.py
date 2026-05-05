import json
import sqlite3
from datetime import datetime
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

CONSOLIDATION_THRESHOLD = 3

try:
    from kairos.core.memory.episodic import get_unconsolidated, mark_consolidated, DB_PATH
except ImportError:
    from kairos.core.memory.episodic import get_unconsolidated, mark_consolidated
    from pathlib import Path
    DB_PATH = Path.home() / ".kairos" / "memory.db"


def _get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize():
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS semantic_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern TEXT NOT NULL,
            classification TEXT NOT NULL,
            confidence REAL,
            examples TEXT,
            created_at TEXT,
            times_applied INTEGER DEFAULT 0,
            times_correct INTEGER DEFAULT 0,
            times_wrong INTEGER DEFAULT 0,
            active INTEGER DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()


def get_patterns(active_only: bool = True) -> list:
    conn = _get_connection()
    query = """
        SELECT id, pattern, classification, confidence, examples,
               times_applied, times_correct, times_wrong
        FROM semantic_patterns
    """
    if active_only:
        query += " WHERE active = 1 AND confidence >= 0.5"
    query += " ORDER BY confidence DESC"
    rows = conn.execute(query).fetchall()
    conn.close()
    return [
        {
            "id": row[0],
            "pattern": row[1],
            "classification": row[2],
            "confidence": row[3],
            "examples": json.loads(row[4]) if row[4] else [],
            "times_applied": row[5],
            "times_correct": row[6],
            "times_wrong": row[7]
        }
        for row in rows
    ]


def decay_pattern(pattern_id: int, reason: str = ""):
    conn = _get_connection()
    conn.execute("""
        UPDATE semantic_patterns
        SET confidence = MAX(0.0, confidence - 0.1),
            times_wrong = times_wrong + 1,
            active = CASE WHEN confidence - 0.1 < 0.3 THEN 0 ELSE 1 END
        WHERE id = ?
    """, (pattern_id,))
    conn.commit()
    conn.close()
    print(f"[Kairos Memory] Pattern {pattern_id} decayed. Reason: {reason or 'wrong output'}")


def reinforce_pattern(pattern_id: int):
    conn = _get_connection()
    conn.execute("""
        UPDATE semantic_patterns
        SET confidence = MIN(0.98, confidence + 0.05),
            times_applied = times_applied + 1,
            times_correct = times_correct + 1
        WHERE id = ?
    """, (pattern_id,))
    conn.commit()
    conn.close()


def _contradicts_existing(pattern: str, classification: str) -> bool:
    patterns = get_patterns()
    new_words = set(pattern.lower().split())
    for existing in patterns:
        if existing["classification"] == classification:
            continue
        existing_words = set(existing["pattern"].lower().split())
        overlap = len(new_words & existing_words) / max(len(new_words), 1)
        if overlap > 0.5:
            print(f"[Kairos Memory] Contradiction blocked: '{pattern[:40]}' vs '{existing['pattern'][:40]}'")
            return True
    return False


def _pattern_exists(pattern: str) -> bool:
    conn = _get_connection()
    rows = conn.execute(
        "SELECT pattern FROM semantic_patterns WHERE active = 1"
    ).fetchall()
    conn.close()
    pattern_words = set(pattern.lower().split())
    for row in rows:
        existing_words = set(row[0].lower().split())
        overlap = len(pattern_words & existing_words) / max(len(pattern_words), 1)
        if overlap > 0.6:
            return True
    return False


def _store_pattern(pattern: str, classification: str, confidence: float, examples: list):
    if _pattern_exists(pattern):
        print(f"[Kairos Memory] Duplicate skipped: {pattern[:50]}")
        return
    if _contradicts_existing(pattern, classification):
        print(f"[Kairos Memory] Contradiction blocked: {pattern[:50]}")
        return
    conn = _get_connection()
    conn.execute("""
        INSERT INTO semantic_patterns
        (pattern, classification, confidence, examples, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (pattern, classification, confidence, json.dumps(examples), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    print(f"[Kairos Memory] Pattern stored: {pattern[:50]}")


def should_consolidate() -> bool:
    from kairos.core.memory.episodic import count_unconsolidated
    return count_unconsolidated() >= CONSOLIDATION_THRESHOLD


def consolidate() -> dict:
    all_episodes = get_unconsolidated(limit=20)
    episodes = [ep for ep in all_episodes if ep.was_correct]
    incorrect = len(all_episodes) - len(episodes)

    if incorrect > 0:
        print(f"[Kairos Memory] Filtered {incorrect} incorrect episodes")

    if len(episodes) < CONSOLIDATION_THRESHOLD:
        return {"status": "skipped", "reason": f"Only {len(episodes)} correct episodes, need {CONSOLIDATION_THRESHOLD}"}

    print(f"[Kairos Memory] Consolidating {len(episodes)} correct episodes...")

    episode_text = "\n".join([
        f"Episode {i}: Task='{ep.task}' Mode={ep.classification_mode} Decision={ep.human_decision}"
        for i, ep in enumerate(episodes, 1)
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""Analyze these correct AI agent experiences and extract 2-4 classification patterns.

{episode_text}

Return ONLY this JSON array (minimum confidence 0.75):
[{{"pattern": "description", "classification": "system1" or "system2", "confidence": 0.75-0.95, "examples": ["ex1", "ex2"]}}]"""

    response = llm.invoke([
        SystemMessage(content="Extract patterns from agent experiences. Return only JSON."),
        HumanMessage(content=prompt)
    ])

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    patterns = json.loads(raw)
    stored = 0
    for p in patterns:
        if p["confidence"] >= 0.75:
            _store_pattern(p["pattern"], p["classification"], p["confidence"], p.get("examples", []))
            stored += 1

    mark_consolidated(all_episodes)
    return {"status": "consolidated", "episodes_processed": len(episodes), "incorrect_filtered": incorrect, "patterns_extracted": stored}
