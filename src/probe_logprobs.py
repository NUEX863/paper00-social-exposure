"""First-token log-probability validation."""
import math
from llm_client import check_response

def valid_logprob(record):
    """Reject absent, nonnumeric or invalid log probabilities explicitly."""
    if not isinstance(record, dict) or not isinstance(record.get("token"), str):
        return False
    value = record.get("logprob")
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value <= 0)

def inspect_logprobs(response):
    """Inspect position zero only; never search later conditioned positions."""
    checks = check_response(response)
    records = response.get("logprobs")
    present = isinstance(records, list) and bool(records)
    valid = present and all(valid_logprob(row) for row in records)
    first = records[0] if valid else {}
    alternatives = first.get("top_logprobs", [])
    alternatives_valid = (isinstance(alternatives, list) and bool(alternatives)
                          and all(valid_logprob(row) for row in alternatives))
    # Keep exact spellings separate: whitespace/case variants are not pooled.
    labels = {label: [row for row in alternatives if row["token"] == label]
              if alternatives_valid else [] for label in ("Yes", "No")}
    both = all(len(rows) == 1 for rows in labels.values())
    content = response.get("message", {}).get("content", "")
    aligned = bool(first.get("token")) and content.startswith(first["token"])
    passed = checks["pass"] and valid and alternatives_valid and both and aligned
    return {
        "status": "PASS" if passed else "REVIEW_REQUIRED",
        "response_checks": checks,
        "logprobs_present": present,
        "token_records_valid": valid,
        "record_count": len(records) if isinstance(records, list) else None,
        "first_recorded_token": first.get("token"),
        "first_token_matches_content_prefix": aligned,
        "first_position_alternatives_valid": alternatives_valid,
        "first_position_alternatives": alternatives,
        "exact_label_entries": labels,
        "both_exact_labels_available": both,
        "adoption_probability": None,
        "limit": "Field-level capability only; no probability normalization, label repair, or adoption estimate. Missing labels are unknown, not zero.",
    }
