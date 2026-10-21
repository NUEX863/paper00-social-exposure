"""Deterministic named random streams for network experiments."""
import hashlib

def seed(key):
    """Use independent, stable named streams rather than Python's random hash."""
    return int.from_bytes(hashlib.sha256(("sprint-v4-20261019:" + key).encode()).digest()[:4], "big")
