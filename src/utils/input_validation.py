import re

MAX_PLAYER_NAME_LENGTH = 64
ALLOWED_PLAYER_NAME_PATTERN = re.compile(r"^[a-zA-Z\s\-']+$")


def sanitize_player_name(player_name: str) -> str:
    if player_name is None:
        raise ValueError("Player name is required")
    if not isinstance(player_name, str):
        raise ValueError("Player name must be a string")

    for i, c in enumerate(player_name):
        code = ord(c)
        if (code <= 31) or (code == 127):
            raise ValueError("Player name contains invalid control characters")

    cleaned = " ".join(player_name.strip().split())

    if not cleaned:
        raise ValueError("Player name is empty after cleaning")
    if len(cleaned) > MAX_PLAYER_NAME_LENGTH:
        raise ValueError(f"Player name must be at most {MAX_PLAYER_NAME_LENGTH} characters")
    if not ALLOWED_PLAYER_NAME_PATTERN.match(cleaned):
        raise ValueError(
            "Player name may only contain letters, spaces, hyphen, and apostrophe"
        )

    return cleaned
