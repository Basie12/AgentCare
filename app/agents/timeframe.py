"""Timeframe parsing for appointment requests.

When a patient says "next week" or "Friday morning", the appointment agent
must honour it rather than grabbing the earliest open slot. This is
deliberately deterministic — no LLM call. Scheduling is the one place where a
hallucinated date is worse than no date, and this runs in microseconds inside
the booking path.

Weeks start Monday (ISO convention): on a Wednesday, "next week" means the
Monday-to-Sunday block that follows, not "seven days from now".
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

WEEKDAYS: dict[str, int] = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

MONTHS: dict[str, int] = {
    name.lower(): num
    for num, name in enumerate(calendar.month_name)
    if name
} | {
    name.lower(): num
    for num, name in enumerate(calendar.month_abbr)
    if name
}

# Time-of-day preference. Boundaries match a normal outpatient clinic day.
PART_OF_DAY_WINDOWS: dict[str, tuple[int, int]] = {
    "morning": (0, 12),
    "afternoon": (12, 17),
    "evening": (17, 24),
}

PART_OF_DAY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("morning", re.compile(r"\b(morning|before noon|early in the day|a\.?m\.?)\b", re.I)),
    ("afternoon", re.compile(r"\b(afternoon|after lunch|midday|p\.?m\.?)\b", re.I)),
    ("evening", re.compile(r"\b(evening|after work|end of the day|late in the day)\b", re.I)),
]

URGENT = re.compile(
    r"\b(asap|as soon as possible|earliest|soonest|first available|any time soon)\b", re.I
)


@dataclass
class Timeframe:
    """A requested scheduling window. All datetimes are naive UTC."""

    start: datetime | None = None
    end: datetime | None = None
    part_of_day: str | None = None
    label: str = "no preference"
    matched_text: str = ""

    @property
    def has_window(self) -> bool:
        return self.start is not None or self.end is not None

    @property
    def specified(self) -> bool:
        return self.has_window or self.part_of_day is not None

    def contains(self, moment: datetime) -> bool:
        if self.start and moment < self.start:
            return False
        if self.end and moment > self.end:
            return False
        if self.part_of_day:
            low, high = PART_OF_DAY_WINDOWS[self.part_of_day]
            if not (low <= moment.hour < high):
                return False
        return True

    def days_ahead(self, now: datetime, default: int = 14) -> int:
        """How far the slot query needs to reach to cover this window."""
        if not self.end:
            return default
        span = (self.end - now).days + 1
        return max(1, min(span, 120))

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "part_of_day": self.part_of_day,
            "matched_text": self.matched_text,
            "specified": self.specified,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _day_start(moment: datetime) -> datetime:
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def _day_end(moment: datetime) -> datetime:
    return moment.replace(hour=23, minute=59, second=59, microsecond=0)


def _week_start(moment: datetime) -> datetime:
    """Monday 00:00 of the week containing `moment`."""
    return _day_start(moment - timedelta(days=moment.weekday()))


def _next_weekday(now: datetime, weekday: int, *, allow_today: bool = False) -> datetime:
    delta = (weekday - now.weekday()) % 7
    if delta == 0 and not allow_today:
        delta = 7
    return _day_start(now + timedelta(days=delta))


def _clamp_future(window: tuple[datetime, datetime], now: datetime) -> tuple[datetime, datetime]:
    """Never return a window that has already passed."""
    start, end = window
    return (max(start, now), end)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def parse_timeframe(text: str, now: datetime | None = None) -> Timeframe:
    """Extract a scheduling window from free text. Never raises."""
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    if not text:
        return Timeframe()

    lowered = text.lower()
    part_of_day = _detect_part_of_day(lowered)

    window = _detect_window(lowered, now)
    if window is None:
        return Timeframe(
            part_of_day=part_of_day,
            label=(f"{part_of_day} (no date given)" if part_of_day else "no preference"),
        )

    start, end, label, matched = window
    start, end = _clamp_future((start, end), now)

    if part_of_day:
        label = f"{label}, {part_of_day}"

    return Timeframe(
        start=start, end=end, part_of_day=part_of_day, label=label, matched_text=matched
    )


def _detect_part_of_day(lowered: str) -> str | None:
    for name, pattern in PART_OF_DAY_PATTERNS:
        if pattern.search(lowered):
            return name
    return None


def _detect_window(
    lowered: str, now: datetime
) -> tuple[datetime, datetime, str, str] | None:
    """Ordered most-specific first — the first match wins."""

    # --- explicit ISO date: 2026-08-03 ---
    match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", lowered)
    if match:
        try:
            day = datetime(int(match[1]), int(match[2]), int(match[3]))
            return _day_start(day), _day_end(day), day.strftime("%a %d %b"), match[0]
        except ValueError:
            pass

    # --- "August 3", "3 August", "Aug 3rd" ---
    month_names = "|".join(sorted(MONTHS, key=len, reverse=True))
    for pattern in (
        rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b",
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({month_names})\b",
    ):
        match = re.search(pattern, lowered)
        if match:
            groups = match.groups()
            month_token, day_token = (
                (groups[0], groups[1]) if groups[0] in MONTHS else (groups[1], groups[0])
            )
            try:
                month = MONTHS[month_token]
                day_num = int(day_token)
                year = now.year if (month, day_num) >= (now.month, now.day) else now.year + 1
                day = datetime(year, month, day_num)
                return _day_start(day), _day_end(day), day.strftime("%a %d %b"), match[0]
            except (ValueError, KeyError):
                pass

    # --- "the week after next" ---
    match = re.search(r"\b(?:the\s+)?week after next\b", lowered)
    if match:
        start = _week_start(now) + timedelta(weeks=2)
        return start, _day_end(start + timedelta(days=6)), "the week after next", match[0]

    # --- "next week" ---
    match = re.search(r"\bnext week\b", lowered)
    if match:
        start = _week_start(now) + timedelta(weeks=1)
        return start, _day_end(start + timedelta(days=6)), "next week", match[0]

    # --- "this week" ---
    match = re.search(r"\bthis week\b", lowered)
    if match:
        end = _week_start(now) + timedelta(days=6)
        return _day_start(now), _day_end(end), "this week", match[0]

    # --- "next month" / "this month" ---
    match = re.search(r"\bnext month\b", lowered)
    if match:
        year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
        last = calendar.monthrange(year, month)[1]
        return (
            datetime(year, month, 1),
            _day_end(datetime(year, month, last)),
            "next month",
            match[0],
        )

    match = re.search(r"\bthis month\b", lowered)
    if match:
        last = calendar.monthrange(now.year, now.month)[1]
        return (
            _day_start(now),
            _day_end(datetime(now.year, now.month, last)),
            "this month",
            match[0],
        )

    # --- "next Friday" (the one in next week) ---
    weekday_names = "|".join(sorted(WEEKDAYS, key=len, reverse=True))
    match = re.search(rf"\bnext ({weekday_names})\b", lowered)
    if match:
        target = _week_start(now) + timedelta(weeks=1, days=WEEKDAYS[match[1]])
        return target, _day_end(target), f"next {match[1].title()}", match[0]

    # --- "in 3 days" / "in two weeks" ---
    match = re.search(
        r"\bin (\d+|a|an|one|two|three|four|five|six)\s+(day|week|month)s?\b", lowered
    )
    if match:
        amount = _word_to_int(match[1])
        unit = match[2]
        target = now + _delta(amount, unit)
        if unit == "day":
            return _day_start(target), _day_end(target), f"in {amount} day(s)", match[0]
        start = _week_start(target) if unit == "week" else _day_start(target)
        return start, _day_end(start + timedelta(days=6)), f"in {amount} {unit}(s)", match[0]

    # --- "within 10 days" / "in the next 2 weeks" / "over the next month" ---
    match = re.search(
        r"\b(?:within|in|over)\s+(?:the\s+)?(?:next\s+)?(\d+|a|an|one|two|three|four)\s+"
        r"(day|week|month)s?\b",
        lowered,
    )
    if match:
        amount = _word_to_int(match[1])
        unit = match[2]
        return (
            _day_start(now),
            _day_end(now + _delta(amount, unit)),
            f"within {amount} {unit}(s)",
            match[0],
        )

    # --- "tomorrow" / "today" ---
    match = re.search(r"\btomorrow\b", lowered)
    if match:
        target = now + timedelta(days=1)
        return _day_start(target), _day_end(target), "tomorrow", match[0]

    match = re.search(r"\btoday\b", lowered)
    if match:
        return _day_start(now), _day_end(now), "today", match[0]

    # --- urgency ---
    match = URGENT.search(lowered)
    if match:
        return _day_start(now), _day_end(now + timedelta(days=3)), "as soon as possible", match[0]

    # --- bare or "on" weekday: "on Friday", "Friday" ---
    match = re.search(rf"\b(?:on\s+|this\s+)?({weekday_names})\b", lowered)
    if match:
        target = _next_weekday(now, WEEKDAYS[match[1]], allow_today=True)
        return target, _day_end(target), match[1].title(), match[0]

    return None


def _word_to_int(token: str) -> int:
    words = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    return words.get(token, int(token) if token.isdigit() else 1)


def _delta(amount: int, unit: str) -> timedelta:
    if unit == "day":
        return timedelta(days=amount)
    if unit == "week":
        return timedelta(weeks=amount)
    return timedelta(days=30 * amount)


# ---------------------------------------------------------------------------
# Slot selection
# ---------------------------------------------------------------------------
def select_slot(slots: list[dict], timeframe: Timeframe) -> tuple[list[dict], str]:
    """Rank slots against the requested timeframe.

    Returns (ordered candidates, explanation). Slots inside the requested
    window come first; the rest are kept as fallbacks so a booking still
    happens when the preference cannot be met, rather than failing outright.
    """
    if not slots:
        return [], "no slots available"

    if not timeframe.specified:
        return slots, "no timeframe requested; earliest available offered"

    def parsed(slot: dict) -> datetime | None:
        try:
            return datetime.fromisoformat(slot["start_time"])
        except (KeyError, ValueError):
            return None

    exact, day_only, rest = [], [], []
    for slot in slots:
        moment = parsed(slot)
        if moment is None:
            rest.append(slot)
        elif timeframe.contains(moment):
            exact.append(slot)
        elif timeframe.has_window and (
            (not timeframe.start or moment >= timeframe.start)
            and (not timeframe.end or moment <= timeframe.end)
        ):
            # Right days, wrong time of day.
            day_only.append(slot)
        else:
            rest.append(slot)

    ordered = exact + day_only + rest

    if exact:
        return ordered, f"matched requested timeframe: {timeframe.label}"
    if day_only:
        return ordered, (
            f"no {timeframe.part_of_day} slot within {timeframe.label}; "
            f"offering the closest available on those days"
        )
    return ordered, f"no availability within {timeframe.label}; offering the earliest instead"
