import sqlite3
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

# Store database in user's home directory so it persists across projects
DB_PATH = Path.home() / ".kairos" / "memory.db"


@dataclass
class Episode:
    task: str
    classification_mode: str      # system1 or system2
    classification_source: str    # heuristic or llm
    classification_confidence: float
    execution_mode: str           # system1, system2, system2_escalated
    output_confidence: float
    human_reviewed: bool
    human_decision: str           # approved, edited, rejected, none
    human_feedback: str
    was_correct: bool             # did human approve without editing?
    timestamp: str = None


def _get_connection():
    """Get SQLite connection, creating database if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn


def initialize():
    """Create tables if they don't exist."""
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            classification_mode TEXT,
            classification_source TEXT,
            classification_confidence REAL,
            execution_mode TEXT,
            output_confidence REAL,
            human_reviewed INTEGER,
            human_decision TEXT,
            human_feedback TEXT,
            was_correct INTEGER,
            timestamp TEXT,
            consolidated INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS consolidation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            episodes_processed INTEGER,
            patterns_extracted INTEGER
        )
    """)
    conn.commit()
    conn.close()


def store(episode: Episode) -> int:
    """Store a new episode. Returns episode ID."""
    conn = _get_connection()
    cursor = conn.execute("""
        INSERT INTO episodes (
            task, classification_mode, classification_source,
            classification_confidence, execution_mode, output_confidence,
            human_reviewed, human_decision, human_feedback,
            was_correct, timestamp, consolidated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
    """, (
        episode.task,
        episode.classification_mode,
        episode.classification_source,
        episode.classification_confidence,
        episode.execution_mode,
        episode.output_confidence,
        1 if episode.human_reviewed else 0,
        episode.human_decision,
        episode.human_feedback,
        1 if episode.was_correct else 0,
        datetime.now().isoformat()
    ))
    conn.commit()
    episode_id = cursor.lastrowid
    conn.close()
    return episode_id


def get_unconsolidated(limit: int = 50) -> list[Episode]:
    """Get episodes not yet consolidated into semantic memory."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT task, classification_mode, classification_source,
               classification_confidence, execution_mode, output_confidence,
               human_reviewed, human_decision, human_feedback,
               was_correct, timestamp
        FROM episodes
        WHERE consolidated = 0
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    episodes = []
    for row in rows:
        episodes.append(Episode(
            task=row[0],
            classification_mode=row[1],
            classification_source=row[2],
            classification_confidence=row[3],
            execution_mode=row[4],
            output_confidence=row[5],
            human_reviewed=bool(row[6]),
            human_decision=row[7],
            human_feedback=row[8],
            was_correct=bool(row[9]),
            timestamp=row[10]
        ))
    return episodes


def mark_consolidated(episodes: list[Episode]):
    """Mark episodes as consolidated so they aren't reprocessed."""
    conn = _get_connection()
    conn.execute("""
        UPDATE episodes SET consolidated = 1
        WHERE consolidated = 0
    """)
    conn.commit()
    conn.close()


def get_similar(task: str, limit: int = 5) -> list[Episode]:
    """
    Retrieve similar past episodes by keyword matching.
    Simple but effective — no embedding needed.
    """
    words = [w.lower() for w in task.split() if len(w) > 4]
    if not words:
        return []

    conn = _get_connection()
    results = []

    for word in words[:5]:
        rows = conn.execute("""
            SELECT task, classification_mode, classification_source,
                   classification_confidence, execution_mode, output_confidence,
                   human_reviewed, human_decision, human_feedback,
                   was_correct, timestamp
            FROM episodes
            WHERE LOWER(task) LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{word}%", limit)).fetchall()

        for row in rows:
            results.append(Episode(
                task=row[0],
                classification_mode=row[1],
                classification_source=row[2],
                classification_confidence=row[3],
                execution_mode=row[4],
                output_confidence=row[5],
                human_reviewed=bool(row[6]),
                human_decision=row[7],
                human_feedback=row[8],
                was_correct=bool(row[9]),
                timestamp=row[10]
            ))

    conn.close()

    # Deduplicate by task
    seen = set()
    unique = []
    for ep in results:
        if ep.task not in seen:
            seen.add(ep.task)
            unique.append(ep)

    return unique[:limit]


def count_unconsolidated() -> int:
    """How many episodes are waiting for consolidation."""
    conn = _get_connection()
    count = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE consolidated = 0"
    ).fetchone()[0]
    conn.close()
    return count


def get_stats() -> dict:
    """Overview of episodic memory."""
    conn = _get_connection()
    total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    correct = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE was_correct = 1"
    ).fetchone()[0]
    human_reviewed = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE human_reviewed = 1"
    ).fetchone()[0]
    rejected = conn.execute(
        "SELECT COUNT(*) FROM episodes WHERE human_decision = 'rejected'"
    ).fetchone()[0]
    conn.close()

    return {
        "total_episodes": total,
        "correct_rate": round(correct / total, 2) if total > 0 else 0,
        "human_review_rate": round(human_reviewed / total, 2) if total > 0 else 0,
        "rejection_rate": round(rejected / total, 2) if total > 0 else 0
    }