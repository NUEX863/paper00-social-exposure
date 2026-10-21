"""Separate response meaning from strict format without altering raw replies."""

import json
from pathlib import Path

POLICY_PATH = Path(__file__).resolve().parents[1] / "configs/era_readout_policy_v1.json"


def classify_response(response, policy=None):
    """Accept only the frozen whitelist; never search prose for a final label."""
    if policy is None:
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    message = response.get("message") or {}
    content = message.get("content")
    text = content.strip() if isinstance(content, str) else ""
    thinking = message.get("thinking") or ""
    eligible = (
        not response.get("error")
        and isinstance(content, str)
        and response.get("done") is True
        and response.get("done_reason") == "stop"
        and not str(thinking).strip()
        and "<think>" not in text and "</think>" not in text
    )
    # An error, truncated reply, or visible reasoning remains invalid under both rules.
    strict = policy["strict_labels"].get(text, "invalid") if eligible else "invalid"
    semantic = policy["semantic_labels"].get(text, "invalid") if eligible else "invalid"
    return {"strict_class": strict, "semantic_class": semantic,
            "strict_format_valid": strict != "invalid", "response_eligible": bool(eligible),
            "punctuation_only_recovery": strict == "invalid" and semantic != "invalid"}
