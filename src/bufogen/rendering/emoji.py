"""Normalize proposed names into portable Slack-style emoji shortcodes."""

import re
import unicodedata


def emoji_shortcode(proposed: str, fallback: str = "drawing") -> str:
    def slug(value):
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
        return re.sub(r"[^a-z0-9_-]+", "-", value.strip().strip(":")).strip("-_")

    name = slug(proposed) or slug(fallback) or "drawing"
    if not name.startswith(("bufo-", "bufo_")) and name != "bufo":
        name = "bufo-" + name
    return f":{name[:64].rstrip('-_')}:"
