"""LangChain Tool로 감싼 가상 캘린더 조회·검증 기능."""

import json

from langchain_core.tools import tool

from models import Plan, PlanningContext
from scheduler import build_free_slots, expand_fixed_events, validate_plan


@tool
def get_virtual_calendar_events(context_json: str) -> list[dict]:
    """계획 기간의 고정 일정을 날짜별 가상 캘린더 이벤트로 조회한다."""
    context = PlanningContext.model_validate_json(context_json)
    if not context.start_date or not context.end_date:
        return []
    return [
        event.model_dump()
        for event in expand_fixed_events(
            context.fixed_events, context.start_date, context.end_date
        )
    ]


@tool
def check_virtual_calendar_conflicts(
    context_json: str,
    plan_json: str,
) -> dict:
    """가상 캘린더의 가용 시간·고정 일정과 계획의 충돌을 검사한다."""
    context = PlanningContext.model_validate_json(context_json)
    plan = Plan.model_validate_json(plan_json)
    free_slots = build_free_slots(
        context.available_slots,
        context.fixed_events,
        context.start_date,
        context.end_date,
    )
    return validate_plan(plan, free_slots).model_dump()
