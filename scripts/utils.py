from itertools import islice
import re

def batched_iterable(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch

def extract_label(text: str) -> int:
    """Extract single digit 0 or 1; fallback to -1 for unknown."""
    m = re.search(r"\b([01])\b", text)
    return int(m.group(1)) if m else -1
