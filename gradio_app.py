"""AI 동적 플래너 Gradio 서비스 화면."""

from datetime import date, datetime, timedelta
from html import escape
from collections import Counter

import gradio as gr
from langchain_core.prompts import ChatPromptTemplate

from app import (
    MAX_QUESTIONS_PER_ROUND, extract_chain, fallback_question, model,
    pair_questions_and_answers, question_chain,
)
from calendar_tools import (
    check_virtual_calendar_conflicts,
    get_virtual_calendar_events,
)
from models import (
    AvailableSlot, FixedEvent, Plan, PlanningContext, PlanUpdate,
    PlanValidation, ScheduleItem,
)
from planner import (
    WEEKDAY_NAMES, apply_suggested_estimates, build_question_needs, create_plan,
    apply_text_constraints, detect_missing_information, extend_unplaced_tasks,
    replan,
)
from prompts import REPLAN_PROMPT
from scheduler import build_free_slots, validate_plan


replan_chain = REPLAN_PROMPT | model.with_structured_output(
    PlanUpdate, method="function_calling"
)


def empty_plan_controls():
    return gr.update(choices=[], value=[]), []


def plan_choices(plan: Plan):
    return [
        f"{item.date} {item.start_time}~{item.end_time} · {item.title} [{item.task_id}]"
        for item in plan.schedule
    ]


def plan_rows(plan: Plan):
    return [
        [item.task_id, item.title, item.date, item.start_time, item.end_time]
        for item in plan.schedule
    ]


def selected_ids(labels):
    result = []
    for label in labels or []:
        if label.endswith("]") and "[" in label:
            result.append(label.rsplit("[", 1)[1][:-1])
    return result


def make_questions(context: PlanningContext):
    needs = build_question_needs(context)[:MAX_QUESTIONS_PER_ROUND]
    if not needs:
        return [], ""
    generated = question_chain.invoke({
        "context": context.model_dump_json(indent=2),
        "question_needs": [need.to_dict() for need in needs],
        "round_number": 1,
        "asked_keys": [],
    })
    need_keys = {need.key for need in needs}
    items = [item for item in generated.questions if item.key in need_keys]
    returned = {item.key for item in items}
    items.extend(fallback_question(need) for need in needs if need.key not in returned)
    blocks = ["### 계획을 만들기 위해 조금만 더 알려주세요"]
    for index, item in enumerate(items, 1):
        block = f"**{index}. {item.question}**"
        if item.example:
            block += f"\n\n> 예: {item.example.removeprefix('예:').strip()}"
        blocks.append(block)
    return [item.model_dump() for item in items], "\n\n---\n\n".join(blocks)


def render_question_data(items):
    blocks = ["### 계획을 만들기 위해 조금만 더 알려주세요"]
    for index, item in enumerate(items, 1):
        block = f"**{index}. {item['question']}**"
        if item.get("example"):
            block += f"\n\n> 예: {item['example'].removeprefix('예:').strip()}"
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


def format_plan_markdown(context, plan, validation, title="생성된 계획"):
    lines = [f"## {title}"]
    strategy_guides = {
        "quick": "가능한 시간 중 빠른 완료를 우선했습니다.",
        "balanced": "계획 기간에 작업을 고르게 분산했습니다.",
        "buffer": "예상치 못한 변경을 위해 마지막 가용일을 비워 두었습니다.",
    }
    lines.append(f"\n> {strategy_guides[context.planning_strategy]} 표에서 직접 수정할 수 있습니다.")
    if plan.schedule:
        completion_date = max(item.date for item in plan.schedule)
        strategy_names = {
            "quick": "빠른 완료", "balanced": "마감일까지 균등 분산",
            "buffer": "마지막 가용일을 비워 두는 여유일 확보",
        }
        lines.append(
            f"\n\n- **계획 기간:** {context.start_date} ~ {context.end_date}"
            f"\n- **배치 전략:** {strategy_names[context.planning_strategy]}"
            f"\n- **예상 완료일:** {completion_date}"
        )
        if completion_date < context.end_date and all(
            not task.is_recurring for task in context.tasks
        ):
            lines.append(
                f"\n\n> 현재 입력한 일회성 작업은 {completion_date}까지 완료할 수 있어 "
                "계획 기간의 모든 날짜에 일정을 만들 필요가 없다고 판단했습니다."
            )

    if context.available_slots:
        lines.append("\n### 반영한 가용 시간")
        for slot in context.available_slots:
            if slot.date:
                label = slot.date
            elif sorted(slot.weekdays) == [0, 1, 2, 3, 4]:
                label = "평일"
            elif sorted(slot.weekdays) == [5, 6]:
                label = "주말"
            else:
                label = "·".join(WEEKDAY_NAMES[day] for day in slot.weekdays)
            if slot.start_time and slot.end_time:
                time_text = f"{slot.start_time}~{slot.end_time}"
            else:
                time_text = f"하루 {slot.available_minutes}분" if slot.available_minutes else "시간 미정"
            lines.append(f"\n- {label} · {time_text}")

    events = get_virtual_calendar_events.invoke({
        "context_json": context.model_dump_json()
    })
    if events:
        lines.append("\n### 반영한 고정 일정")
        for event in events:
            date_text = event["date"]
            if event.get("end_date") and event["end_date"] != event["date"]:
                date_text += f" ~ {event['end_date']}"
            time_text = (
                "종일" if event.get("is_all_day")
                else f"{event['start_time']}~{event['end_time']}"
            )
            lines.append(f"\n- {date_text} {time_text} · {event['title']}")

    lines.append("\n### 실행 일정")
    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    current = None
    for item in sorted(plan.schedule, key=lambda value: (value.date, value.start_time)):
        if item.date != current:
            current = item.date
            day = date.fromisoformat(item.date)
            lines.append(f"\n\n#### {item.date} ({weekdays[day.weekday()]})")
        lines.append(f"\n- **{item.start_time}~{item.end_time}** · {item.title}")
    if not plan.schedule:
        lines.append("\n- 배치된 작업이 없습니다.")

    if plan.warnings:
        lines.append(
            f"\n### 🚨 부분 계획: {len(plan.warnings)}개 작업을 배치하지 못했습니다."
        )
        warning_counts = Counter(plan.warnings)
        for warning, count in warning_counts.items():
            suffix = f" ({count}건)" if count > 1 else ""
            lines.append(f"\n- {warning}{suffix}")
        lines.append(
            "\n\n> 아래 **미배치 해결** 탭에서 여유일 사용, 기간 연장, "
            "추가 가능 시간 등록을 바로 실행할 수 있습니다."
        )
    lines.append(f"\n### 계획 이유\n\n{plan.explanation}")
    if not validation.is_valid:
        result = "충돌 발견"
    elif plan.warnings:
        result = "충돌은 없지만 일부 작업 미배치"
    else:
        result = "모든 작업 배치 완료"
    lines.append(f"\n### 가상 캘린더 검증: {result}")
    lines.extend(f"\n- {error}" for error in validation.errors)
    return "".join(lines)


def weekly_calendar_html(context, plan):
    if not context or not context.start_date or not context.end_date:
        return "<div class='calendar-empty'>계획을 생성하면 주간 캘린더가 표시됩니다.</div>"
    start, end = date.fromisoformat(context.start_date), date.fromisoformat(context.end_date)
    week_start = start - timedelta(days=start.weekday())
    task_map = {}
    for item in plan.schedule:
        task_map.setdefault(item.date, []).append(item)
    fixed = get_virtual_calendar_events.invoke({"context_json": context.model_dump_json()})
    fixed_map = {}
    for event in fixed:
        event_start = date.fromisoformat(event["date"])
        event_end = date.fromisoformat(event.get("end_date") or event["date"])
        cursor = event_start
        while cursor <= event_end:
            fixed_map.setdefault(cursor.isoformat(), []).append(event)
            cursor += timedelta(days=1)
    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    weeks = []
    while week_start <= end:
        days = []
        for offset in range(7):
            day = week_start + timedelta(days=offset)
            classes = "calendar-day"
            if day < start or day > end:
                classes += " outside"
            if day.weekday() in context.unavailable_weekdays:
                classes += " unavailable"
            cards = []
            if day.weekday() in context.unavailable_weekdays:
                cards.append("<div class='event-card unavailable-card'>계획 불가</div>")
            for event in fixed_map.get(day.isoformat(), []):
                time_text = (
                    "종일" if event.get("is_all_day")
                    else f"{event['start_time']}~{event['end_time']}"
                )
                cards.append(
                    f"<div class='event-card fixed-card'><b>{escape(event['title'])}</b>"
                    f"<small>{time_text}</small></div>"
                )
            for item in sorted(task_map.get(day.isoformat(), []), key=lambda value: value.start_time):
                cards.append(
                    f"<div class='event-card task-card'><b>{escape(item.title)}</b>"
                    f"<small>{item.start_time}~{item.end_time}</small></div>"
                )
            days.append(
                f"<div class='{classes}'><div class='day-header'>{weekdays[offset]} "
                f"<span>{day.month}/{day.day}</span></div>{''.join(cards)}</div>"
            )
        week_end = week_start + timedelta(days=6)
        weeks.append(
            f"<section class='calendar-week'><h3>{week_start:%m/%d} ~ {week_end:%m/%d}</h3>"
            f"<div class='week-grid'>{''.join(days)}</div></section>"
        )
        week_start += timedelta(days=7)
    warning = ""
    if plan.warnings:
        warning = (
            f"<div class='calendar-warning'><b>🚨 부분 계획</b><br>"
            f"{len(plan.warnings)}개 작업이 아직 미배치 상태입니다. "
            "아래 미배치 해결 탭에서 추가 가능 시간을 등록해 주세요.</div>"
        )
    return warning + "<div class='calendar-wrap'>" + "".join(weeks) + "</div>"


def generate_plan_output(context):
    plan, _ = create_plan(context, model)
    tool_result = check_virtual_calendar_conflicts.invoke({
        "context_json": context.model_dump_json(),
        "plan_json": plan.model_dump_json(),
    })
    validation = PlanValidation.model_validate(tool_result)
    return plan, validation, format_plan_markdown(context, plan, validation)


def analyze_request(user_input, strategy):
    empty_choices, empty_rows = empty_plan_controls()
    if not user_input.strip():
        return "요청을 입력해 주세요.", "", "", "", {}, [], "", {}, empty_choices, empty_rows
    context = extract_chain.invoke({
        "current_date": date.today().isoformat(), "user_input": user_input,
    })
    context = apply_text_constraints(context, user_input, date.today())
    context.planning_strategy = strategy
    context.missing_information = detect_missing_information(context)
    questions, markdown = make_questions(context)
    history = f"[최초 요청]\n{user_input}"
    if questions:
        return (
            "입력을 분석했습니다. 아래 질문에 답해주세요.", markdown, "", "",
            context.model_dump(), questions, history, {}, empty_choices, empty_rows,
        )
    context = apply_suggested_estimates(context)
    plan, _, output = generate_plan_output(context)
    return (
        "추가 질문 없이 계획을 생성했습니다.", "", output,
        weekly_calendar_html(context, plan),
        context.model_dump(), [], history, plan.model_dump(),
        gr.update(choices=plan_choices(plan), value=[]), plan_rows(plan),
    )


def apply_answers(answer_text, context_data, question_data, history, strategy):
    empty_choices, empty_rows = empty_plan_controls()
    if not context_data:
        return "먼저 요청 분석을 눌러주세요.", "", "", "", {}, [], history, {}, empty_choices, empty_rows
    if question_data and not answer_text.strip():
        return (
            "질문에 답해주세요.", render_question_data(question_data), "", "",
            context_data, question_data, history, {}, empty_choices, empty_rows,
        )
    context = PlanningContext.model_validate(context_data)
    paired = pair_questions_and_answers(
        [item["question"] for item in question_data], answer_text
    )
    history = f"{history}\n\n[질문과 답변]\n{paired}"
    merged = f"""{
        history
    }

[현재까지 구조화된 정보]
{context.model_dump_json(indent=2)}

최근 답변을 우선해 PlanningContext를 다시 작성한다.
답하지 않은 정보는 만들지 않고 충돌하지 않는 기존 정보는 유지한다.
질문과 답변은 순서대로 연결한다."""
    updated = extract_chain.invoke({
        "current_date": date.today().isoformat(), "user_input": merged,
    })
    updated = apply_text_constraints(updated, merged, date.today())
    updated.planning_strategy = strategy
    updated.missing_information = detect_missing_information(updated)
    questions, markdown = make_questions(updated)
    if questions:
        return (
            "아직 필요한 정보가 있습니다.", markdown, "", "", updated.model_dump(),
            questions, history, {}, empty_choices, empty_rows,
        )
    updated = apply_suggested_estimates(updated)
    plan, _, output = generate_plan_output(updated)
    return (
        "계획 생성과 검증을 완료했습니다.", "", output,
        weekly_calendar_html(updated, plan), updated.model_dump(),
        [], history, plan.model_dump(),
        gr.update(choices=plan_choices(plan), value=[]), plan_rows(plan),
    )


def build_comparison(previous, updated):
    old = {item.task_id: item for item in previous.schedule}
    new = {item.task_id: item for item in updated.schedule}
    blocks = []
    icons = {"유지": "✅", "이동": "🔄", "추가": "➕", "미배치": "⚠️"}
    for task_id in dict.fromkeys([*old, *new]):
        before, after = old.get(task_id), new.get(task_id)
        before_text = (
            f"{before.date} {before.start_time}~{before.end_time}" if before else "-"
        )
        after_text = (
            f"{after.date} {after.start_time}~{after.end_time}" if after else "-"
        )
        if before and after:
            state = "유지" if before_text == after_text else "이동"
            title = after.title
        elif after:
            state, title = "추가", after.title
        else:
            state, title = "미배치", before.title
        blocks.append(
            f"**{icons[state]} {state} · {title}**  \n"
            f"기존: {before_text} → 변경: {after_text}"
        )
    return "\n\n---\n\n".join(blocks)


def apply_replan(
    selected_labels, event_type, event_title,
    event_start_date, event_end_date, event_start_at, event_end_at,
    note, context_data, plan_data,
):
    if not context_data or not plan_data:
        return (
            "먼저 최초 계획을 생성해 주세요.", "", "", "",
            context_data, plan_data, gr.update(), [],
        )
    context = PlanningContext.model_validate(context_data)
    previous = Plan.model_validate(plan_data)
    missed_ids = selected_ids(selected_labels)
    new_events = []

    def unchanged_result(message):
        current_markdown = format_plan_markdown(
            context,
            previous,
            validate_plan(
                previous,
                build_free_slots(
                    context.available_slots, context.fixed_events,
                    context.start_date, context.end_date,
                ),
            ),
        )
        return (
            message, current_markdown, "",
            weekly_calendar_html(context, previous),
            context_data, plan_data, gr.update(), plan_rows(previous),
        )

    has_event_input = any([
        event_title, event_start_date, event_end_date, event_start_at, event_end_at,
    ])
    if has_event_input and event_type == "all_day":
        if not all([event_title, event_start_date, event_end_date]):
            return unchanged_result(
                "종일·기간 일정을 추가하려면 이름·시작 날짜·종료 날짜를 모두 입력해 주세요."
            )
        if event_end_date < event_start_date:
            return unchanged_result("종료 날짜는 시작 날짜보다 빠를 수 없습니다.")
        if (
            event_start_date.date() < date.fromisoformat(context.start_date)
            or event_end_date.date() > date.fromisoformat(context.end_date)
        ):
            return unchanged_result(
                f"새 일정은 현재 계획 기간({context.start_date}~{context.end_date}) 안에서만 "
                "추가할 수 있습니다. 먼저 미배치 해결 탭에서 계획 기간을 연장하거나 "
                "기간 안의 날짜를 선택해 주세요."
            )
        new_events.append(FixedEvent(
            id=f"ui-event-{len(context.fixed_events)+1}",
            title=event_title,
            date=event_start_date.date().isoformat(),
            end_date=event_end_date.date().isoformat(),
            is_all_day=True,
        ))
    elif has_event_input and event_type == "timed":
        if not all([event_title, event_start_at, event_end_at]):
            return unchanged_result(
                "시간 지정 일정을 추가하려면 이름·시작 일시·종료 일시를 모두 입력해 주세요."
            )
        if event_end_at <= event_start_at:
            return unchanged_result("새 일정의 종료 일시는 시작 일시보다 늦어야 합니다.")
        if (
            event_start_at.date() < date.fromisoformat(context.start_date)
            or event_end_at.date() > date.fromisoformat(context.end_date)
        ):
            return unchanged_result(
                f"새 일정은 현재 계획 기간({context.start_date}~{context.end_date}) 안에서만 "
                "추가할 수 있습니다. 먼저 미배치 해결 탭에서 계획 기간을 연장하거나 "
                "기간 안의 일시를 선택해 주세요."
            )
        new_events.append(FixedEvent(
            id=f"ui-event-{len(context.fixed_events)+1}",
            title=event_title,
            date=event_start_at.date().isoformat(),
            end_date=(event_end_at.date().isoformat() if event_end_at.date() != event_start_at.date() else None),
            is_all_day=False,
            start_time=event_start_at.strftime("%H:%M"),
            end_time=event_end_at.strftime("%H:%M"),
        ))

    missed_ids = list(dict.fromkeys(missed_ids))
    if not missed_ids:
        return unchanged_result(
            "체크박스에서 완료하지 못한 작업을 하나 이상 선택해 주세요."
        )

    updated, validation = replan(
        context, previous, missed_ids, new_events, model
    )
    display_context = context.model_copy(deep=True)
    display_context.fixed_events.extend(new_events)
    comparison = "## 기존 계획과 변경 계획 비교\n\n" + build_comparison(
        previous, updated
    )
    output = comparison + "\n\n" + format_plan_markdown(
        display_context, updated, validation, "재계획 결과"
    )
    if note.strip():
        output += f"\n\n### 사용자가 남긴 재계획 메모\n\n{note.strip()}"
    if updated.warnings:
        status = (
            f"재계획했지만 가용 시간이 부족해 {len(updated.warnings)}개 작업을 "
            "배치하지 못했습니다. 아래 해결 방법을 확인해 주세요."
        )
    else:
        status = "선택한 미완료 작업과 새 일정을 반영했습니다."
    return (
        status, output, output, weekly_calendar_html(display_context, updated),
        display_context.model_dump(), updated.model_dump(),
        gr.update(choices=plan_choices(updated), value=[]),
        plan_rows(updated),
    )


def apply_plan_edits(rows, context_data, plan_data):
    if not context_data or not plan_data:
        return "먼저 계획을 생성해 주세요.", "", "", plan_data, gr.update(), rows
    values = rows.values.tolist() if hasattr(rows, "values") else rows
    try:
        schedule = []
        for task_id, title, day, start, end in values:
            start_dt = datetime.fromisoformat(f"{str(day)[:10]}T{start}")
            end_dt = datetime.fromisoformat(f"{str(day)[:10]}T{end}")
            minutes = int((end_dt - start_dt).total_seconds() // 60)
            if minutes <= 0:
                raise ValueError(f"{title}: 종료 시간은 시작 시간보다 늦어야 합니다.")
            schedule.append(ScheduleItem(
                task_id=str(task_id), title=str(title), date=str(day)[:10],
                start_time=str(start), end_time=str(end), minutes=minutes,
            ))
    except Exception as error:
        return f"수정한 표를 확인해 주세요: {error}", "", "", plan_data, gr.update(), rows

    context = PlanningContext.model_validate(context_data)
    previous = Plan.model_validate(plan_data)
    edited = previous.model_copy(deep=True)
    edited.schedule = sorted(schedule, key=lambda item: (item.date, item.start_time))
    free = build_free_slots(
        context.available_slots, context.fixed_events,
        context.start_date, context.end_date,
    )
    validation = validate_plan(edited, free)
    output = "## 수정 전후 비교\n\n" + build_comparison(previous, edited)
    output += "\n\n" + format_plan_markdown(
        context, edited, validation, "사용자가 수정한 계획"
    )
    status = (
        "수정 내용을 저장했고 검증을 통과했습니다."
        if validation.is_valid else
        "수정 내용은 저장했지만 충돌이 있습니다. 검증 결과를 확인해 주세요."
    )
    return (
        status, output, weekly_calendar_html(context, edited), edited.model_dump(),
        gr.update(choices=plan_choices(edited), value=[]), plan_rows(edited),
    )


def toggle_event_type(event_type):
    is_all_day = event_type == "all_day"
    return (
        gr.update(visible=is_all_day),
        gr.update(visible=is_all_day),
        gr.update(visible=not is_all_day),
        gr.update(visible=not is_all_day),
    )


def resolution_result(context, message):
    plan, _, markdown = generate_plan_output(context)
    remaining = len(plan.warnings)
    status = (
        f"{message} 모든 작업을 배치했습니다."
        if remaining == 0 else
        f"{message} 다시 계산했지만 {remaining}개 작업은 아직 미배치 상태입니다."
    )
    return (
        status, markdown, markdown, weekly_calendar_html(context, plan),
        context.model_dump(), plan.model_dump(),
        gr.update(choices=plan_choices(plan), value=[]), plan_rows(plan),
    )


def use_reserved_day(context_data, plan_data):
    if not context_data or not plan_data:
        return "먼저 계획을 생성해 주세요.", "", "", "", context_data, plan_data, gr.update(), []
    context = PlanningContext.model_validate(context_data)
    context.planning_strategy = "balanced"
    return resolution_result(context, "비워 두었던 마지막 가용일을 사용해")


def extend_unplaced_period(new_end_at, context_data, plan_data):
    if not context_data or not plan_data:
        return "먼저 계획을 생성해 주세요.", "", "", "", context_data, plan_data, gr.update(), []
    if not new_end_at:
        return "새 계획 종료 날짜를 선택해 주세요.", "", "", "", context_data, plan_data, gr.update(), []
    context = PlanningContext.model_validate(context_data)
    previous_plan = Plan.model_validate(plan_data)
    try:
        updated_context, updated_plan, validation = extend_unplaced_tasks(
            context,
            previous_plan,
            new_end_at.date().isoformat(),
            model,
        )
    except ValueError as error:
        return (
            str(error), "", "", weekly_calendar_html(context, previous_plan),
            context_data, plan_data, gr.update(), plan_rows(previous_plan),
        )
    markdown = format_plan_markdown(updated_context, updated_plan, validation)
    remaining = len(updated_plan.unscheduled_tasks)
    status = (
        "기존 일정과 반복 횟수는 유지하고 미배치 작업만 연장 구간에 배치했습니다."
        if remaining == 0 else
        f"연장 구간에 다시 배치했지만 {remaining}개 작업은 아직 미배치 상태입니다."
    )
    return (
        status, markdown, markdown,
        weekly_calendar_html(updated_context, updated_plan),
        updated_context.model_dump(), updated_plan.model_dump(),
        gr.update(choices=plan_choices(updated_plan), value=[]),
        plan_rows(updated_plan),
    )


def add_extra_availability(extra_start, extra_end, context_data, plan_data):
    if not context_data or not plan_data:
        return "먼저 계획을 생성해 주세요.", "", "", "", context_data, plan_data, gr.update(), []
    if not extra_start or not extra_end:
        return "추가로 가능한 시작·종료 일시를 모두 선택해 주세요.", "", "", "", context_data, plan_data, gr.update(), []
    if extra_end <= extra_start:
        return "추가 가능 시간의 종료는 시작보다 늦어야 합니다.", "", "", "", context_data, plan_data, gr.update(), []
    if extra_start.date() != extra_end.date():
        return "추가 가능 시간은 같은 날짜 안에서 선택해 주세요.", "", "", "", context_data, plan_data, gr.update(), []
    context = PlanningContext.model_validate(context_data)
    context.available_slots.append(AvailableSlot(
        id=f"extra-{len(context.available_slots)+1}",
        date=extra_start.date().isoformat(),
        start_time=extra_start.strftime("%H:%M"),
        end_time=extra_end.strftime("%H:%M"),
    ))
    return resolution_result(context, "추가 가능 시간을 반영해")


CSS = """
.gradio-container {max-width: 1180px !important; margin: auto !important;}
.hero {padding: 24px; border-radius: 18px; background: linear-gradient(135deg,#eef2ff,#f5f3ff);}
.hero, .hero h1, .hero p {color: #1f2937 !important;}
.section-card {border: 1px solid #e5e7eb; border-radius: 16px; padding: 10px;}
#status-box {border-left: 4px solid #6366f1; padding-left: 14px;}
.calendar-wrap {display:flex; flex-direction:column; gap:20px;}
.calendar-warning {margin-bottom:14px; padding:14px; border-radius:12px; background:#fef2f2; color:#991b1b; border:1px solid #fecaca;}
.calendar-week h3 {margin:0 0 8px; font-size:15px; color:#4f46e5;}
.week-grid {display:grid; grid-template-columns:repeat(7,minmax(110px,1fr)); gap:7px; overflow-x:auto;}
.calendar-day {min-height:130px; padding:8px; border:1px solid #e5e7eb; border-radius:12px; background:#fff;}
.calendar-day.outside {opacity:.35;}
.calendar-day.unavailable {background:#fef2f2;}
.day-header {font-weight:700; margin-bottom:7px; display:flex; justify-content:space-between;}
.day-header span {font-weight:400; color:#6b7280;}
.event-card {padding:7px; border-radius:8px; margin:5px 0; font-size:12px;}
.event-card small {display:block; margin-top:3px;}
.task-card {background:#ede9fe; color:#4c1d95; border-left:3px solid #7c3aed;}
.fixed-card {background:#f3f4f6; color:#374151; border-left:3px solid #6b7280;}
.unavailable-card {background:#fee2e2; color:#991b1b;}
@media (prefers-color-scheme: dark) {
  .hero {background: linear-gradient(135deg,#1e1b4b,#312e81);}
  .hero, .hero h1, .hero p {color: #f9fafb !important;}
  .section-card {border-color: #374151;}
  .calendar-day {background:#111827; border-color:#374151;}
  .calendar-day.unavailable {background:#3f1d25;}
  .task-card {background:#312e81; color:#ede9fe;}
  .fixed-card {background:#374151; color:#f3f4f6;}
}
@media (max-width: 800px) {.week-grid {grid-template-columns:repeat(7,150px);}}
"""
THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="violet")

with gr.Blocks(title="AI 동적 플래너") as demo:
    gr.Markdown(
        "# AI 동적 플래너\n해야 할 일을 말하면 필요한 조건만 확인하고 실행 가능한 일정으로 정리합니다.",
        elem_classes=["hero"],
    )
    context_state, question_state = gr.State({}), gr.State([])
    history_state, plan_state = gr.State(""), gr.State({})

    with gr.Row():
        with gr.Column(scale=5, elem_classes=["section-card"]):
            gr.Markdown("## 1. 계획 요청")
            request_input = gr.Textbox(
                label="어떤 계획을 세우고 싶나요?", lines=6,
                placeholder="예: 이번 주 일요일까지 LangChain 프로젝트를 완성하고 싶어...",
            )
            strategy_input = gr.Radio(
                choices=[
                    ("빠르게 끝내기", "quick"),
                    ("마감일까지 균등 분산", "balanced"),
                    ("마지막 가용일을 비워 두기", "buffer"),
                ],
                value="quick",
                label="계획 방식",
            )
            analyze_button = gr.Button("요청 분석", variant="primary")
            status_output = gr.Markdown(elem_id="status-box")
            question_output = gr.Markdown()
            answer_input = gr.Textbox(
                label="추가 질문 답변", lines=5,
                placeholder="질문 번호 순서대로 한 줄씩 답해주세요.",
            )
            create_button = gr.Button("답변 반영하고 계획 만들기", variant="primary")

        with gr.Column(scale=7, elem_classes=["section-card"]):
            gr.Markdown("## 2. 계획 결과")
            with gr.Tabs():
                with gr.Tab("주간 캘린더"):
                    calendar_output = gr.HTML(
                        "<div class='calendar-empty'>계획을 생성하면 주간 캘린더가 표시됩니다.</div>"
                    )
                with gr.Tab("날짜별 상세"):
                    plan_output = gr.Markdown()

    with gr.Tabs():
        with gr.Tab("미배치 해결"):
            gr.Markdown("""
            ### 미배치 작업을 바로 다시 계획하기
            계획 결과에 `부분 계획` 경고가 있을 때 아래 방법 중 하나를 실행하세요.
            실행 후 계획과 주간 캘린더가 즉시 갱신됩니다.
            """)
            use_buffer_button = gr.Button(
                "비워 둔 여유일 사용", variant="primary"
            )
            with gr.Group():
                gr.Markdown(
                    "**기간을 늘려 해결하려면**  \n"
                    "기존 일정과 반복 작업 횟수는 바꾸지 않고, 현재 미배치 작업만 "
                    "기존 종료일 다음 날부터 새 종료일까지 배치합니다."
                )
                with gr.Row():
                    extended_end_date = gr.DateTime(
                        label="새 계획 종료 날짜",
                        include_time=False, type="datetime", timezone="Asia/Seoul",
                    )
                    extend_button = gr.Button(
                        "미배치 작업만 연장 구간에 배치", variant="secondary"
                    )
            with gr.Group():
                gr.Markdown("**특정 날짜에 추가로 시간을 낼 수 있다면**")
                with gr.Row():
                    extra_start = gr.DateTime(
                        label="추가 가능 시작 일시",
                        include_time=True, type="datetime", timezone="Asia/Seoul",
                    )
                    extra_end = gr.DateTime(
                        label="추가 가능 종료 일시",
                        include_time=True, type="datetime", timezone="Asia/Seoul",
                    )
                    add_time_button = gr.Button("추가 시간 반영")
            resolution_output = gr.Markdown()

        with gr.Tab("재계획"):
            gr.Markdown("계획대로 끝내지 못한 작업을 선택하세요. 새 일정 입력은 선택 사항입니다.")
            missed_selector = gr.CheckboxGroup(
                label="계획대로 완료하지 못한 작업", choices=[],
                info="체크한 작업과 그 이후 일정만 다시 배치합니다.",
            )
            event_type = gr.Radio(
                choices=[
                    ("종일·기간 일정", "all_day"),
                    ("시간 지정 일정", "timed"),
                ],
                value="all_day",
                label="새 일정 유형(선택)",
                info="출장·여행처럼 날짜 범위 전체가 불가능하면 종일·기간 일정을 선택합니다.",
            )
            with gr.Row():
                event_title = gr.Textbox(
                    label="새 일정 이름(선택)",
                    placeholder="예: 출장, 회의",
                    info="새 일정을 추가할 때만 날짜 또는 일시와 함께 입력합니다.",
                )
                event_start_date = gr.DateTime(
                    label="시작 날짜(선택)", include_time=False,
                    type="datetime", timezone="Asia/Seoul",
                )
                event_end_date = gr.DateTime(
                    label="종료 날짜(선택)", include_time=False,
                    type="datetime", timezone="Asia/Seoul",
                    info="시작일부터 종료일까지 모든 가용 시간을 막습니다.",
                )
                event_start_at = gr.DateTime(
                    label="시작 일시(선택)", include_time=True,
                    type="datetime", timezone="Asia/Seoul", visible=False,
                )
                event_end_at = gr.DateTime(
                    label="종료 일시(선택)", include_time=True,
                    type="datetime", timezone="Asia/Seoul",
                    visible=False,
                )
            replan_note = gr.Textbox(
                label="재계획 메모(선택)",
                placeholder="예: 병원 일정 때문에 이번 주에는 시간이 부족함",
                info="메모는 결과에 함께 표시되며 일정이나 작업으로 자동 변환되지 않습니다.",
            )
            replan_button = gr.Button("변경 사항 반영해 재계획", variant="primary")
            replan_output = gr.Markdown()

        with gr.Tab("계획 직접 수정"):
            gr.Markdown("날짜와 시간을 수정한 뒤 저장하면 충돌을 다시 검사합니다.")
            plan_editor = gr.Dataframe(
                headers=["task_id", "작업", "날짜", "시작", "종료"],
                datatype=["str", "str", "str", "str", "str"],
                interactive=True,
                row_count=(0, "dynamic"),
            )
            edit_button = gr.Button("수정 내용 저장 및 검증")
            edit_output = gr.Markdown()

    common_outputs = [
        status_output, question_output, plan_output, calendar_output,
        context_state, question_state, history_state, plan_state,
        missed_selector, plan_editor,
    ]
    analyze_button.click(
        analyze_request, [request_input, strategy_input], common_outputs
    )
    create_button.click(
        apply_answers,
        [answer_input, context_state, question_state, history_state, strategy_input],
        common_outputs,
    )
    replan_button.click(
        apply_replan,
        [
            missed_selector, event_type, event_title,
            event_start_date, event_end_date, event_start_at, event_end_at,
            replan_note, context_state, plan_state,
        ],
        [
            status_output, plan_output, replan_output, calendar_output,
            context_state, plan_state, missed_selector, plan_editor,
        ],
    )
    event_type.change(
        toggle_event_type,
        inputs=[event_type],
        outputs=[event_start_date, event_end_date, event_start_at, event_end_at],
    )
    edit_button.click(
        apply_plan_edits,
        [plan_editor, context_state, plan_state],
        [status_output, edit_output, calendar_output, plan_state, missed_selector, plan_editor],
    )
    resolution_outputs = [
        status_output, plan_output, resolution_output, calendar_output,
        context_state, plan_state, missed_selector, plan_editor,
    ]
    use_buffer_button.click(
        use_reserved_day,
        [context_state, plan_state],
        resolution_outputs,
    )
    extend_button.click(
        extend_unplaced_period,
        [extended_end_date, context_state, plan_state],
        resolution_outputs,
    )
    add_time_button.click(
        add_extra_availability,
        [extra_start, extra_end, context_state, plan_state],
        resolution_outputs,
    )


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
