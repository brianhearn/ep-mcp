"""Age-based freshness for ExpertPack atoms (ExpertPack Core 4.2 §Freshness).

Two independent signals, both separate from the index-drift ``stale`` flag:

* ``age_stale`` — ``verified_at`` is older than the pack's
  ``manifest.freshness.refresh_cycle`` (or a per-file ``refresh`` override).
* ``expired`` — a volatile file's ``expires_at`` is in the past.

``stale`` (index drift: the file changed since it was indexed) is unchanged.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime

_ISO_DURATION_RE = re.compile(
    r"^P(?:(?P<y>\d+)Y)?(?:(?P<m>\d+)M)?(?:(?P<w>\d+)W)?(?:(?P<d>\d+)D)?$",
    re.IGNORECASE,
)
_NAMED_CYCLES = {
    "daily": 1, "weekly": 7, "biweekly": 14, "monthly": 30, "quarterly": 91,
    "semiannual": 182, "semiannually": 182, "biannual": 182,
    "annual": 365, "annually": 365, "yearly": 365,
}


def parse_duration_days(value: object) -> int | None:
    """Parse an ISO 8601 date duration (P90D, P3M, P1Y, P2W) or a named cycle to days.

    Returns None for unparseable/non-time cycles such as ``per-release``.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in _NAMED_CYCLES:
        return _NAMED_CYCLES[text.lower()]
    m = _ISO_DURATION_RE.match(text)
    if not m or not any(m.groupdict().values()):
        return None
    g = {k: int(v) if v else 0 for k, v in m.groupdict().items()}
    days = g["y"] * 365 + g["m"] * 30 + g["w"] * 7 + g["d"]
    return days or None


def parse_date(value: object) -> date | None:
    """Parse YYYY-MM-DD or ISO datetime into a date. Returns None when invalid."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def compute_freshness(
    verified_at: object,
    refresh_cycle: object,
    expires_at: object = None,
    file_refresh: object = None,
    today: date | None = None,
) -> dict:
    """Return ``{verified_age_days, age_stale, expired, expires_at}``.

    ``age_stale`` and ``expired`` are None when the inputs needed to decide are missing.
    """
    today = today or datetime.now(UTC).date()
    verified = parse_date(verified_at)
    age_days = (today - verified).days if verified else None
    cycle_days = parse_duration_days(file_refresh) or parse_duration_days(refresh_cycle)
    age_stale = None
    if age_days is not None and cycle_days is not None:
        age_stale = age_days > cycle_days
    exp = parse_date(expires_at)
    expired = (exp < today) if exp else None
    return {
        "verified_age_days": age_days,
        "age_stale": age_stale,
        "expired": expired,
        "expires_at": exp.isoformat() if exp else None,
    }


def freshness_for_file(pack, pack_file, today: date | None = None) -> dict:
    """Compute freshness fields for a PackFile within its pack."""
    cycle = pack.freshness.refresh_cycle if pack.freshness else None
    return compute_freshness(
        pack_file.provenance.verified_at,
        cycle,
        expires_at=pack_file.expires_at,
        file_refresh=pack_file.refresh,
        today=today,
    )
