# utils.py
import re

DELIMITER = r"---\s*Now here is the actual question you need to answer\."

def extract_answer(response: str) -> str:
    # Split off the examples
    parts = re.split(DELIMITER, response, flags=re.IGNORECASE)

    # If delimiter exists, only search after it
    target = parts[-1]

    # Prefer explicit Answer:
    match = re.search(r"Answer:\s*(.+)", target, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: first non-empty line after assistant output
    lines = [l.strip() for l in target.splitlines() if l.strip()]
    if lines:
        return lines[-1]

    return target.strip()