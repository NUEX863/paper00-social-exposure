"""Native Ollama requests and response validation."""
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class OllamaClient:
    """Send JSON requests directly to a configured Ollama endpoint."""

    def __init__(self, base_url, timeout=180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Local requests must not inherit an unrelated Internet proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, route, payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + route, data=body,
            headers={"Content-Type": "application/json"},
        )
        with self.opener.open(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

def git_output(*args):
    """Return unavailable Git metadata as null."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def check_response(response):
    """Validate a raw response without extracting or repairing an answer."""
    message = response.get("message", {})
    content = message.get("content", "")
    thinking = message.get("thinking", "") or ""
    clean_answer = content.strip() if isinstance(content, str) else ""
    binary_ok = clean_answer in {"Yes", "No"}
    thinking_ok = (
        not str(thinking).strip()
        and "<think>" not in clean_answer
        and "</think>" not in clean_answer
    )
    completed = response.get("done") is True and response.get("done_reason") == "stop"
    return {
        "answer_exact_yes_no": binary_ok,
        "no_visible_thinking": thinking_ok,
        "finished_without_truncation": completed,
        "done_reason": response.get("done_reason"),
        "pass": binary_ok and thinking_ok and completed,
        "limit": "One response cannot validate all prompts or prove absence of internal reasoning.",
    }
