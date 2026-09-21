from calendar import monthrange
from datetime import date, datetime, time, timedelta

from models import AvailableSlot, FixedEvent, Plan, PlanValidation, ScheduleItem, Task


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _add_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 + months
    year, month_zero_based = divmod(month_index, 12)
    month = month_zero_based + 1
    day = min(value.day, monthrange(year, month)[1])
    return date(year, month, day)


def expand_fixed_event(
    event: FixedEvent,
    plan_start: str,
    plan_end: str,
) -> list[FixedEvent]:
    """고정 일정의 반복 규칙을 계획 기간 안의 실제 날짜 목록으로 확장한다."""
    start = _parse_date(plan_start)
    end = _parse_date(plan_end)

    if end < start:
        raise ValueError("plan_end는 plan_start보다 빠를 수 없습니다.")

    if event.recurrence is None:
        if event.date is None:
            raise ValueError(f"단일 일정 '{event.title}'에 날짜가 없습니다.")
        event_date = _parse_date(event.date)
        return [event] if start <= event_date <= end else []

    rule = event.recurrence
    if rule.until is not None:
        end = min(end, _parse_date(rule.until))

    anchor = max(start, _parse_date(event.date)) if event.date else start
    occurrences: list[date] = []

    def append_if_allowed(candidate: date) -> bool:
        if candidate < start or candidate > end:
            return False
        occurrences.append(candidate)
        return rule.count is not None and len(occurrences) >= rule.count

    if rule.frequency == "daily":
        current = anchor
        while current <= end:
            if append_if_allowed(current):
                break
            current += timedelta(days=rule.interval)

    elif rule.frequency == "weekly":
        if not rule.weekdays:
            raise ValueError(f"주간 반복 일정 '{event.title}'에 요일이 없습니다.")

        current = anchor
        while current <= end:
            days_from_anchor = (current - anchor).days
            week_index = days_from_anchor // 7
            if week_index % rule.interval == 0 and current.weekday() in rule.weekdays:
                if append_if_allowed(current):
                    break
            current += timedelta(days=1)

    elif rule.frequency == "monthly":
        if rule.day_of_month is None:
            raise ValueError(f"월간 반복 일정 '{event.title}'에 날짜가 없습니다.")

        month_cursor = date(anchor.year, anchor.month, 1)
        while month_cursor <= end:
            last_day = monthrange(month_cursor.year, month_cursor.month)[1]
            if rule.day_of_month <= last_day:
                candidate = date(
                    month_cursor.year,
                    month_cursor.month,
                    rule.day_of_month,
                )
                if candidate >= anchor and append_if_allowed(candidate):
                    break
            month_cursor = _add_months(month_cursor, rule.interval)

    return [
        FixedEvent(
            id=f"{event.id}-{occurrence.isoformat()}",
            title=event.title,
            date=occurrence.isoformat(),
            start_time=event.start_time,
            end_time=event.end_time,
            recurrence=None,
        )
        for occurrence in occurrences
    ]


def expand_fixed_events(
    events: list[FixedEvent],
    plan_start: str,
    plan_end: str,
) -> list[FixedEvent]:
    """여러 고정 일정을 날짜순으로 확장한다."""
    expanded = [
        occurrence
        for event in events
        for occurrence in expand_fixed_event(event, plan_start, plan_end)
    ]
    return sorted(expanded, key=lambda item: (item.date or "", item.start_time or ""))


def _to_datetime(day: date, value: str) -> datetime:
    return datetime.combine(day, time.fromisoformat(value))


def build_free_slots(
    available_slots: list[AvailableSlot], fixed_events: list[FixedEvent],
    plan_start: str, plan_end: str,
) -> list[tuple[datetime, datetime]]:
    """가용 시간을 날짜별로 펼치고 고정 일정과 겹치는 부분을 제거한다."""
    start, end = _parse_date(plan_start), _parse_date(plan_end)
    event_ranges = []
    for event in expand_fixed_events(fixed_events, plan_start, plan_end):
        if event.date and event.start_time and event.end_time:
            start_day = _parse_date(event.date)
            end_day = _parse_date(event.end_date or event.date)
            event_ranges.append((
                _to_datetime(start_day, event.start_time),
                _to_datetime(end_day, event.end_time),
            ))

    free: list[tuple[datetime, datetime]] = []
    current = start
    while current <= end:
        for slot in available_slots:
            applies = slot.date == current.isoformat() or (
                slot.date is None and current.weekday() in slot.weekdays
            )
            if not applies or not slot.start_time or not slot.end_time:
                continue
            pieces = [(_to_datetime(current, slot.start_time), _to_datetime(current, slot.end_time))]
            for busy_start, busy_end in event_ranges:
                next_pieces = []
                for piece_start, piece_end in pieces:
                    if busy_end <= piece_start or busy_start >= piece_end:
                        next_pieces.append((piece_start, piece_end))
                    else:
                        if piece_start < busy_start:
                            next_pieces.append((piece_start, busy_start))
                        if busy_end < piece_end:
                            next_pieces.append((busy_end, piece_end))
                pieces = next_pieces
            free.extend(piece for piece in pieces if piece[1] > piece[0])
        current += timedelta(days=1)
    return sorted(free)


def expand_task_instances(tasks: list[Task], plan_start: str, plan_end: str) -> list[Task]:
    """반복 작업을 주 단위 횟수만큼 독립적인 배치 단위로 펼친다."""
    start, end = _parse_date(plan_start), _parse_date(plan_end)
    instances: list[Task] = []
    for task in tasks:
        if not task.is_recurring:
            instances.append(task.model_copy(deep=True))
            continue
        week_start, number = start, 1
        while week_start <= end:
            week_end = min(week_start + timedelta(days=6), end)
            count = min(task.frequency_per_week or 1, (week_end - week_start).days + 1)
            for _ in range(count):
                instance = task.model_copy(deep=True)
                instance.id = f"{task.id}-{number}"
                instance.title = f"{task.title} {number}회차"
                instance.available_from = week_start.isoformat()
                if instance.deadline is None or _parse_date(instance.deadline) > week_end:
                    instance.deadline = week_end.isoformat()
                instances.append(instance)
                number += 1
            week_start += timedelta(days=7)
    return instances


def allocate_tasks(
    tasks: list[Task],
    free_slots: list[tuple[datetime, datetime]],
    strategy: str = "quick",
):
    """LLM이 정한 순서의 작업을 실제 빈 시간에 배치한다."""
    slots = [[start, end] for start, end in free_slots]
    schedule, warnings = [], []
    recurring_days: set[tuple[str, date]] = set()
    plan_days = sorted({slot[0].date() for slot in slots})
    usable_days = plan_days[:-1] if strategy == "buffer" and len(plan_days) > 1 else plan_days
    for task_index, task in enumerate(tasks):
        minutes = task.estimated_minutes or 60
        candidates = slots
        if strategy in {"balanced", "buffer"} and usable_days:
            denominator = max(len(tasks) - 1, 1)
            target_index = round(task_index * (len(usable_days) - 1) / denominator)
            target_day = usable_days[target_index]
            candidates = sorted(
                slots,
                key=lambda slot: (
                    slot[0].date() < target_day,
                    abs((slot[0].date() - target_day).days),
                    slot[0],
                ),
            )
        for slot in candidates:
            if strategy == "buffer" and usable_days and slot[0].date() not in usable_days:
                continue
            if task.available_from and slot[0].date() < _parse_date(task.available_from):
                continue
            if task.deadline and slot[0].date() > _parse_date(task.deadline):
                continue
            parts = task.id.rsplit("-", 1)
            recurring_group = parts[0] if len(parts) == 2 and parts[1].isdigit() else None
            if recurring_group and (recurring_group, slot[0].date()) in recurring_days:
                continue
            if int((slot[1] - slot[0]).total_seconds() // 60) < minutes:
                continue
            item_start = slot[0]
            item_end = item_start + timedelta(minutes=minutes)
            schedule.append(ScheduleItem(
                task_id=task.id, title=task.title, date=item_start.date().isoformat(),
                start_time=item_start.strftime("%H:%M"), end_time=item_end.strftime("%H:%M"),
                minutes=minutes,
            ))
            slot[0] = item_end
            if recurring_group:
                recurring_days.add((recurring_group, item_start.date()))
            break
        else:
            warnings.append(f"'{task.title}'을 배치할 충분한 연속 시간이 없습니다.")
    return schedule, warnings


def validate_plan(plan: Plan, free_slots: list[tuple[datetime, datetime]]) -> PlanValidation:
    """계획이 가용 시간 안에 있고 서로 겹치지 않는지 검사한다."""
    errors, ranges = [], []
    for item in plan.schedule:
        day = _parse_date(item.date)
        start, end = _to_datetime(day, item.start_time), _to_datetime(day, item.end_time)
        if int((end - start).total_seconds() // 60) != item.minutes:
            errors.append(f"{item.title}: 시간 길이와 minutes가 다릅니다.")
        if not any(a <= start and end <= b for a, b in free_slots):
            errors.append(f"{item.title}: 가용 시간 밖이거나 고정 일정과 겹칩니다.")
        ranges.append((start, end, item.title))
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            errors.append(f"{previous[2]}와 {current[2]} 일정이 겹칩니다.")
    return PlanValidation(is_valid=not errors, errors=errors, warnings=list(plan.warnings))
