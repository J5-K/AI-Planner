from dataclasses import asdict, dataclass
from datetime import date, timedelta
import re

from models import AvailableSlot, Plan, PlanningContext, PlanningDecision
from prompts import PLAN_PROMPT
from scheduler import allocate_tasks, build_free_slots, expand_task_instances, validate_plan


WEEKDAY_NAMES = {
    0: "월요일",
    1: "화요일",
    2: "수요일",
    3: "목요일",
    4: "금요일",
    5: "토요일",
    6: "일요일",
}

KOREAN_WEEKDAYS = {name: number for number, name in WEEKDAY_NAMES.items()}

AVAILABILITY_DAY_GROUPS = {
    "평일": [0, 1, 2, 3, 4],
    "주말": [5, 6],
    **{name: [number] for name, number in KOREAN_WEEKDAYS.items()},
}


def _to_24_hour(period: str | None, hour: int, minute: int) -> str | None:
    """한국어 오전·오후 표현을 HH:MM으로 변환한다."""
    if minute > 59 or hour > 23:
        return None
    if period == "오전":
        hour = 0 if hour == 12 else hour
    elif period in {"오후", "저녁", "밤"}:
        if hour < 12:
            hour += 12
    return f"{hour:02d}:{minute:02d}"


def _restore_explicit_availability(
    context: PlanningContext,
    source_text: str,
) -> None:
    """LLM 병합에서 빠진 명시적 평일·주말·요일 시간대를 원문으로 복원한다."""
    time_token = (
        r"(?:(오전|오후|저녁|밤)\s*)?"
        r"(\d{1,2})(?::(\d{2}))?\s*시?"
    )
    for label, weekdays in AVAILABILITY_DAY_GROUPS.items():
        pattern = re.compile(
            rf"{label}[^\n,;]{{0,35}}?{time_token}"
            rf"\s*(?:부터|에서|~|～|-)\s*{time_token}(?:\s*까지)?",
            re.IGNORECASE,
        )
        matches = list(pattern.finditer(source_text))
        if not matches:
            continue
        match = matches[-1]
        start_period, start_hour, start_minute, end_period, end_hour, end_minute = (
            match.groups()
        )
        end_period = end_period or start_period
        start_time = _to_24_hour(
            start_period, int(start_hour), int(start_minute or 0)
        )
        end_time = _to_24_hour(
            end_period, int(end_hour), int(end_minute or 0)
        )
        allowed_days = [
            day for day in weekdays if day not in context.unavailable_weekdays
        ]
        if not start_time or not end_time or not allowed_days:
            continue
        existing = next(
            (
                slot for slot in context.available_slots
                if slot.date is None and sorted(slot.weekdays) == allowed_days
            ),
            None,
        )
        if existing:
            existing.start_time = start_time
            existing.end_time = end_time
            existing.available_minutes = None
        else:
            context.available_slots.append(AvailableSlot(
                id=f"text-availability-{label}",
                weekdays=allowed_days,
                start_time=start_time,
                end_time=end_time,
            ))


def apply_text_constraints(
    context: PlanningContext,
    source_text: str,
    reference_date: date | None = None,
) -> PlanningContext:
    """요일 불가와 이번 주·다음 주 날짜를 원문 기준으로 재검증한다."""
    updated = context.model_copy(deep=True)
    reference = reference_date or date.today()
    unavailable = set(updated.unavailable_weekdays)
    for name, number in KOREAN_WEEKDAYS.items():
        patterns = [
            rf"{name}.{{0,12}}(?:안\s*돼|안\s*됨|불가|불가능|어려워|제외)",
            rf"(?:안\s*돼|불가|불가능).{{0,12}}{name}",
        ]
        if any(re.search(pattern, source_text, re.IGNORECASE) for pattern in patterns):
            unavailable.add(number)
    updated.unavailable_weekdays = sorted(unavailable)
    if unavailable:
        for slot in updated.available_slots:
            slot.weekdays = [day for day in slot.weekdays if day not in unavailable]
        for event in updated.fixed_events:
            if event.recurrence and "불가" in event.title:
                event.recurrence.weekdays = [
                    day for day in event.recurrence.weekdays if day in unavailable
                ]

    _restore_explicit_availability(updated, source_text)

    relative_deadline = re.search(
        r"(이번\s*주|다음\s*주)\s*"
        r"(월요일|화요일|수요일|목요일|금요일|토요일|일요일)\s*까지",
        source_text,
    )
    if relative_deadline:
        week_text, weekday_text = relative_deadline.groups()
        monday = reference - timedelta(days=reference.weekday())
        week_offset = 7 if re.sub(r"\s+", "", week_text) == "다음주" else 0
        target = monday + timedelta(
            days=week_offset + KOREAN_WEEKDAYS[weekday_text]
        )
        old_end_date = updated.end_date
        updated.end_date = target.isoformat()
        if updated.start_date is None:
            updated.start_date = reference.isoformat()
        for task in updated.tasks:
            if task.deadline == old_end_date:
                task.deadline = target.isoformat()
    return updated


@dataclass
class QuestionNeed:
    key: str
    topic: str
    details: list[str]
    priority: int

    def to_dict(self) -> dict:
        return asdict(self)


def _describe_event(event) -> str:
    if event.recurrence is None:
        return f"{event.date or '날짜 미정'} {event.title}"

    rule = event.recurrence
    if rule.frequency == "weekly" and rule.weekdays:
        weekdays = "·".join(WEEKDAY_NAMES[day] for day in rule.weekdays)
        interval_text = "매주" if rule.interval == 1 else f"{rule.interval}주마다"
        return f"{interval_text} {weekdays} {event.title}"
    if rule.frequency == "daily":
        interval_text = "매일" if rule.interval == 1 else f"{rule.interval}일마다"
        return f"{interval_text} {event.title}"
    if rule.frequency == "monthly" and rule.day_of_month is not None:
        interval_text = "매달" if rule.interval == 1 else f"{rule.interval}개월마다"
        return f"{interval_text} {rule.day_of_month}일 {event.title}"
    return f"반복 일정 {event.title}"


def describe_available_slot(slot) -> str:
    """내부 슬롯 ID 대신 사용자에게 보여줄 자연스러운 이름을 만든다."""
    if slot.date:
        return f"{slot.date} 가용 시간"
    if slot.weekdays == [0, 1, 2, 3, 4]:
        return "평일"
    if slot.weekdays == [5, 6]:
        return "주말"
    if slot.weekdays:
        return "·".join(WEEKDAY_NAMES[day] for day in slot.weekdays)
    return "가용 시간"


def is_valid_time(value: str | None) -> bool:
    """HH:MM 형식이며 실제 시각으로 유효한지 확인한다."""
    if value is None or re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value) is None:
        return False
    return True


def build_question_needs(context: PlanningContext) -> list[QuestionNeed]:
    """현재 상태에서 꼭 확인해야 하는 내용을 질문 주제 단위로 묶는다."""
    needs: list[QuestionNeed] = []

    if context.start_date is None or context.end_date is None:
        missing_period_parts = []
        if context.start_date is None:
            missing_period_parts.append("시작일")
        if context.end_date is None:
            missing_period_parts.append("종료일")
        needs.append(QuestionNeed(
            key="plan_period",
            topic="전체 계획 기간",
            details=missing_period_parts,
            priority=2,
        ))

    if not context.tasks:
        needs.append(QuestionNeed(
            key="tasks",
            topic="목표별 실행 작업",
            details=["각 목표를 위해 실제로 할 활동이나 작업"],
            priority=1,
        ))

    if not context.available_slots:
        needs.append(QuestionNeed(
            key="availability",
            topic="평일과 주말의 가용 시간",
            details=["평일 시간대", "주말 시간대"],
            priority=1,
        ))
    else:
        incomplete_slots = []
        for slot in context.available_slots:
            slot_label = describe_available_slot(slot)
            missing_parts = []
            if slot.date is None and not slot.weekdays:
                missing_parts.append("날짜 또는 요일")
            if not is_valid_time(slot.start_time):
                missing_parts.append("시작 시간")
            if not is_valid_time(slot.end_time):
                missing_parts.append("종료 시간")
            if missing_parts:
                incomplete_slots.append(
                    f"{slot_label}: {', '.join(missing_parts)}"
                )

        if incomplete_slots:
            needs.append(QuestionNeed(
                key="availability_detail",
                topic="불완전한 가용 시간대",
                details=incomplete_slots,
                priority=1,
            ))

    incomplete_events = []
    for event in context.fixed_events:
        event_label = _describe_event(event)
        missing_parts = []
        if event.recurrence is None and event.date is None:
            missing_parts.append("날짜")
        if event.recurrence is not None:
            rule = event.recurrence
            if rule.frequency == "weekly" and not rule.weekdays:
                missing_parts.append("반복 요일")
            if rule.frequency == "monthly" and rule.day_of_month is None:
                missing_parts.append("매월 반복 날짜")
        if not event.is_all_day:
            if not is_valid_time(event.start_time):
                missing_parts.append("시작 시간")
            if not is_valid_time(event.end_time):
                missing_parts.append("종료 시간")
        if missing_parts:
            incomplete_events.append(
                f"{event_label}: {', '.join(missing_parts)}"
            )

    if incomplete_events:
        needs.append(QuestionNeed(
            key="fixed_event_details",
            topic="고정 일정 정보",
            details=incomplete_events,
            priority=1,
        ))

    tasks_without_frequency = [
        task.title
        for task in context.tasks
        if task.is_recurring and task.frequency_per_week is None
    ]
    if tasks_without_frequency:
        needs.append(QuestionNeed(
            key="task_frequency",
            topic="반복 작업의 주간 횟수",
            details=tasks_without_frequency,
            priority=2,
        ))

    return sorted(needs, key=lambda item: item.priority)


def detect_missing_information(context: PlanningContext) -> list[str]:
    """하위 호환용: 필수 질문 주제를 짧은 목록으로 반환한다."""
    return [need.topic for need in build_question_needs(context)]


DEFAULT_ESTIMATED_MINUTES = 60


def apply_suggested_estimates(context: PlanningContext) -> PlanningContext:
    """시간을 모르는 작업에 수정 가능한 임시 예상 시간을 적용한다."""
    updated = context.model_copy(deep=True)

    for task in updated.tasks:
        if task.estimated_minutes is not None:
            if task.estimate_source == "unknown":
                task.estimate_source = "user"
            continue

        task.estimated_minutes = DEFAULT_ESTIMATED_MINUTES
        task.estimate_source = "suggested"

    return updated


def create_plan(context: PlanningContext, model):
    """LLM의 순서 판단과 Python의 시간 계산을 결합해 계획을 만든다."""
    if not context.start_date or not context.end_date:
        raise ValueError("계획 시작일과 종료일이 필요합니다.")
    prepared = apply_suggested_estimates(context)
    free_slots = build_free_slots(
        prepared.available_slots, prepared.fixed_events,
        prepared.start_date, prepared.end_date,
    )
    instance_slots = free_slots
    if prepared.planning_strategy == "buffer" and free_slots:
        last_available_date = max(slot[0].date() for slot in free_slots)
        instance_slots = [
            slot for slot in free_slots if slot[0].date() != last_available_date
        ]
    instances = expand_task_instances(
        prepared.tasks,
        prepared.start_date,
        prepared.end_date,
        available_dates={slot[0].date() for slot in instance_slots},
    )
    chain = PLAN_PROMPT | model.with_structured_output(PlanningDecision, method="function_calling")
    decision = chain.invoke({
        "goal": prepared.goal,
        "period": f"{prepared.start_date}~{prepared.end_date}",
        "tasks": [task.model_dump() for task in instances],
        "events": [event.model_dump() for event in prepared.fixed_events],
    })
    task_by_id = {task.id: task for task in instances}
    ordered = [task_by_id[task_id] for task_id in decision.ordered_task_ids if task_id in task_by_id]
    ordered_ids = {task.id for task in ordered}
    ordered.extend(task for task in instances if task.id not in ordered_ids)
    recurring_tasks = sorted(
        (task for task in ordered if task.is_recurring),
        key=lambda task: (task.available_from or "", task.title, task.id),
    )
    one_time_tasks = [task for task in ordered if not task.is_recurring]
    # 마감이 있는 핵심 일회성 작업을 먼저 확보하고 반복 루틴은 남는 시간에 배치한다.
    ordered = one_time_tasks + recurring_tasks
    schedule, warnings, unscheduled = allocate_tasks(
        ordered, free_slots, strategy=prepared.planning_strategy
    )
    schedule.sort(key=lambda item: (item.date, item.start_time, item.end_time))
    plan = Plan(
        schedule=schedule,
        warnings=warnings,
        unscheduled_tasks=unscheduled,
        explanation=decision.explanation,
    )
    return plan, validate_plan(plan, free_slots)


def replan(context, previous_plan, missed_task_ids, new_fixed_events, model):
    """기존 완료 일정은 유지하고 실패일 이후의 일정을 다시 배치한다."""
    updated = context.model_copy(deep=True)
    updated.fixed_events.extend(new_fixed_events)
    missed_ids = set(missed_task_ids)
    missed_items = [
        item for item in previous_plan.schedule if item.task_id in missed_ids
    ]
    if not missed_items:
        raise ValueError("기존 계획에서 미완료 작업을 찾지 못했습니다.")

    failed_date = min(item.date for item in missed_items)
    failed_retry_date = date.fromisoformat(failed_date) + timedelta(days=1)
    new_event_start_dates = [
        date.fromisoformat(event.date)
        for event in new_fixed_events
        if event.date is not None
    ]
    replan_start = min(
        [failed_retry_date, *new_event_start_dates]
    )
    preserved = [
        item.model_copy(deep=True)
        for item in previous_plan.schedule
        if date.fromisoformat(item.date) < replan_start
        and item.task_id not in missed_ids
    ]
    affected = [
        item for item in previous_plan.schedule
        if item.task_id in missed_ids
        or date.fromisoformat(item.date) >= replan_start
    ]

    base_tasks = {task.id: task for task in updated.tasks}
    remaining_tasks = []
    for item in affected:
        parts = item.task_id.rsplit("-", 1)
        base_id = parts[0] if len(parts) == 2 and parts[1].isdigit() else item.task_id
        base = base_tasks.get(base_id)
        if base is None:
            continue
        task = base.model_copy(deep=True)
        task.id = item.task_id
        task.title = item.title
        task.estimated_minutes = item.minutes
        task.is_recurring = False
        task.frequency_per_week = None
        task.available_from = None
        remaining_tasks.append(task)

    updated.tasks = remaining_tasks
    updated.start_date = replan_start.isoformat()
    replanned, validation = create_plan(updated, model)
    replanned.schedule = sorted(
        preserved + replanned.schedule,
        key=lambda item: (item.date, item.start_time, item.end_time),
    )
    replanned.explanation = (
        f"{len(preserved)}개의 기존 일정은 유지하고, "
        f"미완료 작업과 이후 일정을 {updated.start_date}부터 다시 배치했습니다. "
        f"{replanned.explanation}"
    )
    return replanned, validation


def extend_unplaced_tasks(
    context: PlanningContext,
    previous_plan: Plan,
    new_end_date: str,
    model,
):
    """기존 일정과 반복 횟수는 유지하고 미배치 작업만 연장 구간에 배치한다."""
    if not context.end_date:
        raise ValueError("기존 계획 종료일이 필요합니다.")
    old_end = date.fromisoformat(context.end_date)
    new_end = date.fromisoformat(new_end_date)
    if new_end <= old_end:
        raise ValueError("새 종료일은 기존 계획 종료일보다 늦어야 합니다.")
    if not previous_plan.unscheduled_tasks:
        raise ValueError("연장 구간에 배치할 미배치 작업이 없습니다.")

    extension_context = context.model_copy(deep=True)
    extension_context.start_date = (old_end + timedelta(days=1)).isoformat()
    extension_context.end_date = new_end.isoformat()
    for event in extension_context.fixed_events:
        if (
            event.recurrence
            and event.recurrence.until
            and date.fromisoformat(event.recurrence.until) <= old_end
        ):
            event.recurrence.until = new_end.isoformat()
    extension_context.planning_strategy = "quick"
    extension_context.tasks = []
    for original in previous_plan.unscheduled_tasks:
        task = original.model_copy(deep=True)
        task.is_recurring = False
        task.frequency_per_week = None
        task.available_from = None
        task.deadline = new_end.isoformat()
        extension_context.tasks.append(task)

    extension_plan, _ = create_plan(extension_context, model)
    combined_context = context.model_copy(deep=True)
    combined_context.end_date = new_end.isoformat()
    for event in combined_context.fixed_events:
        if (
            event.recurrence
            and event.recurrence.until
            and date.fromisoformat(event.recurrence.until) <= old_end
        ):
            event.recurrence.until = new_end.isoformat()
    combined_plan = Plan(
        schedule=sorted(
            previous_plan.schedule + extension_plan.schedule,
            key=lambda item: (item.date, item.start_time, item.end_time),
        ),
        warnings=extension_plan.warnings,
        unscheduled_tasks=extension_plan.unscheduled_tasks,
        explanation=(
            f"기존 일정과 반복 작업 횟수는 유지하고, 미배치 작업만 "
            f"{extension_context.start_date}~{new_end_date} 구간에 추가 배치했습니다."
        ),
    )
    free_slots = build_free_slots(
        combined_context.available_slots,
        combined_context.fixed_events,
        combined_context.start_date,
        combined_context.end_date,
    )
    return combined_context, combined_plan, validate_plan(combined_plan, free_slots)
