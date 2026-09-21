from dataclasses import asdict, dataclass
from datetime import date, timedelta
import re

from models import Plan, PlanningContext, PlanningDecision
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

KOREAN_WEEKDAYS = {
    "월요일": 0, "화요일": 1, "수요일": 2, "목요일": 3,
    "금요일": 4, "토요일": 5, "일요일": 6,
}


def apply_text_constraints(context: PlanningContext, source_text: str) -> PlanningContext:
    """LLM이 혼동하기 쉬운 '요일 불가' 표현을 원문으로 다시 검증한다."""
    updated = context.model_copy(deep=True)
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


DEFAULT_TASK_MINUTES = {
    "운동": 60,
    "걷기": 60,
    "기록": 15,
    "주제": 120,
    "조사": 120,
    "계획": 60,
    "개발": 120,
    "구현": 120,
    "ui": 120,
    "테스트": 60,
    "검토": 60,
}


def apply_suggested_estimates(context: PlanningContext) -> PlanningContext:
    """시간을 모르는 작업에 수정 가능한 임시 예상 시간을 적용한다."""
    updated = context.model_copy(deep=True)

    for task in updated.tasks:
        if task.estimated_minutes is not None:
            if task.estimate_source == "unknown":
                task.estimate_source = "user"
            continue

        normalized_title = task.title.lower()
        task.estimated_minutes = next(
            (
                minutes
                for keyword, minutes in DEFAULT_TASK_MINUTES.items()
                if keyword in normalized_title
            ),
            60,
        )
        task.estimate_source = "suggested"

    return updated


def create_plan(context: PlanningContext, model):
    """LLM의 순서 판단과 Python의 시간 계산을 결합해 계획을 만든다."""
    if not context.start_date or not context.end_date:
        raise ValueError("계획 시작일과 종료일이 필요합니다.")
    prepared = apply_suggested_estimates(context)
    instances = expand_task_instances(prepared.tasks, prepared.start_date, prepared.end_date)
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
    free_slots = build_free_slots(
        prepared.available_slots, prepared.fixed_events, prepared.start_date, prepared.end_date
    )
    schedule, warnings = allocate_tasks(
        ordered, free_slots, strategy=prepared.planning_strategy
    )
    schedule.sort(key=lambda item: (item.date, item.start_time, item.end_time))
    plan = Plan(schedule=schedule, warnings=warnings, explanation=decision.explanation)
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
    preserved = [
        item.model_copy(deep=True)
        for item in previous_plan.schedule
        if item.date <= failed_date and item.task_id not in missed_ids
    ]
    affected = [
        item for item in previous_plan.schedule
        if item.task_id in missed_ids or item.date > failed_date
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
    updated.start_date = (
        date.fromisoformat(failed_date) + timedelta(days=1)
    ).isoformat()
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
