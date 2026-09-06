from __future__ import annotations

import re
from dataclasses import dataclass

SENSITIVE = re.compile(
    r"\b(password|passcode|api[ _-]?key|secret|token|credit card|ssn|social security|"
    r"medical|diagnosis|cancer|health condition|religion|religious|political|ethnicity|"
    r"sexual orientation|precise address)\b",
    re.IGNORECASE,
)
TEMPORARY = re.compile(
    r"\b(today|tomorrow|yesterday|this trip|this weekend|weather|forecast|reservation)\b",
    re.IGNORECASE,
)
DURABLE = re.compile(
    r"\b(vegetarian|vegan|diet|pace|walking|walk|museum|art|history|outdoor|start(?:ing)? time|usually|prefer|preference)\b",
    re.IGNORECASE,
)
MEMORY_RELEVANT = re.compile(
    r"\b(plan|itinerary|attraction|visit|things to do|day in|recommend|remember about me|preference)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MemoryDecision:
    eligible: bool
    reason: str
    category: str = "preference"


def should_search_memory(message: str) -> bool:
    return bool(MEMORY_RELEVANT.search(message)) and "do not use memory" not in message.lower()


def evaluate_memory_write(message: str) -> MemoryDecision:
    text = message.strip()
    if not re.search(r"\bremember\b", text, re.IGNORECASE):
        return MemoryDecision(False, "Explicit 'remember' authorization is required")
    if SENSITIVE.search(text):
        return MemoryDecision(False, "Sensitive information must not be stored")
    if TEMPORARY.search(text):
        return MemoryDecision(False, "Temporary trip facts must not be stored")
    if not DURABLE.search(text):
        return MemoryDecision(False, "No supported durable travel preference was found")
    return MemoryDecision(True, "Explicit durable preference")


def is_forget_request(message: str) -> bool:
    return bool(re.search(r"\b(forget|delete)\b", message, re.IGNORECASE))


def extract_forget_query(message: str) -> str:
    return re.sub(r"^.*?\b(?:forget|delete)\b", "", message, flags=re.IGNORECASE).strip(" .:;")


def contains_prompt_injection(text: str) -> bool:
    patterns = ("ignore previous", "system prompt", "developer message", "call the tool")
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


def is_safe_memory_evidence(text: str) -> bool:
    return not SENSITIVE.search(text) and not contains_prompt_injection(text)
