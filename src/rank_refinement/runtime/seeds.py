"""Independent streams based on semantic keys, not Python's randomized hash."""
import hashlib


def keyed_seed(seed, *parts):
    value = str(seed) + "\x1f" + "\x1f".join(str(x) for x in parts)
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")
