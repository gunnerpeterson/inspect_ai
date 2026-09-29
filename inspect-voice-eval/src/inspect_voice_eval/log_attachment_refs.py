from __future__ import annotations

from inspect_ai._util.hash import mm3_hash
from inspect_ai.log._condense import ATTACHMENT_PROTOCOL


def attachment_ref_for_text(text: str) -> str:
    """Return the attachment reference Inspect log condensation uses for text."""
    return f"{ATTACHMENT_PROTOCOL}{mm3_hash(text)}"
