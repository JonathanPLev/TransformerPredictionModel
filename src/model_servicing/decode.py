import pandas as pd
from dataclasses import dataclass


@dataclass
class DecodeResult:
    success: bool
    df: pd.DataFrame | None = None
    person_id: int | None = None
    error: str | None = None
    status_code: int = 200


def decode_and_fetch(user_string: str) -> DecodeResult:
    try:
        person_id = int(user_string)
    except ValueError:
        return DecodeResult(
            success=False,
            error="Player not found",
            status_code=404,
        )

    # Temporary: return empty dataframe placeholder
    df = pd.DataFrame()
    return DecodeResult(
        success=True,
        df=df,
        person_id=person_id,
    )