"""Common text rules, independent of topic quality and collection source."""
import ast
import re
import unicodedata

URL = re.compile(r"https?://[A-Za-z0-9:/?#\[\]@!$&'()*+,;=%._~-]+", re.IGNORECASE)
PLACEHOLDERS = ("网页链接", "微博视频", "展开全文", "显示全文")


def unwrap_body(text):
    value = unicodedata.normalize("NFKC", str(text)).strip()
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError, TypeError):
            parsed = None
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], str):
            value = parsed[0]
        else:
            value = value[1:-1]
    return re.sub(r"\s+", " ", value).strip()


def clean_body(text):
    value = URL.sub(" ", unwrap_body(text))
    for placeholder in PLACEHOLDERS:
        value = value.replace(placeholder, " ")
    # Preserve punctuation, negation, hashtags and meaningful content for the full-text view.
    return re.sub(r"\s+", " ", value).strip()


def comparison_text(text):
    return re.sub(r"\s+", "", clean_body(text))
