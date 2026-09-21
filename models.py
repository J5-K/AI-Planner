from typing import Literal
from pydantic import BaseModel, Field


class Task(BaseModel):
    id: str = Field(description="작업을 구별하기 위한 고유 ID")
    title: str = Field(description="할 일의 이름")

    estimated_minutes: int | None = Field(
        default=None,
        description="작업 1회당 예상 소요 시간"
    )

    estimate_source: Literal["user", "suggested", "unknown"] = Field(
        default="unknown",
        description="예상 시간이 사용자 입력인지 AI 임시 제안인지 표시"
    )

    deadline: str | None = Field(
        default=None,
        description="YYYY-MM-DD 형식의 마감일"
    )

    available_from: str | None = Field(
        default=None,
        description="이 작업을 배치할 수 있는 최초 날짜. 내부 일정 계산에도 사용"
    )

    priority: int | None = Field(
        default=None,
        description="1이 가장 높은 우선순위"
    )

    splittable: bool | None = Field(
        default=None,
        description="작업을 여러 시간대로 나눌 수 있는지 여부"
    )

    is_recurring: bool = Field(
        default=False,
        description="계획 기간 동안 반복 수행하는 작업인지 여부"
    )

    frequency_per_week: int | None = Field(
        default=None,
        description="반복 작업의 주간 수행 횟수. 일회성 작업이면 null"
    )


class RecurrenceRule(BaseModel):
    frequency: Literal["daily", "weekly", "monthly"] = Field(
        description="반복 단위"
    )
    interval: int = Field(
        default=1,
        ge=1,
        description="몇 단위마다 반복하는지. 매주는 1, 격주는 2"
    )
    weekdays: list[int] = Field(
        default_factory=list,
        description="반복 요일. 월요일=0부터 일요일=6까지"
    )
    day_of_month: int | None = Field(
        default=None,
        ge=1,
        le=31,
        description="매월 반복되는 날짜"
    )
    until: str | None = Field(
        default=None,
        description="반복 종료일. YYYY-MM-DD 형식"
    )
    count: int | None = Field(
        default=None,
        ge=1,
        description="반복 횟수"
    )


class FixedEvent(BaseModel):
    id: str = Field(description="고정 일정을 구별하기 위한 고유 ID")
    title: str
    date: str | None = Field(
        default=None,
        description="단일 일정 날짜 또는 반복 일정의 시작 기준일"
    )
    end_date: str | None = Field(
        default=None,
        description="여러 날 일정의 종료 날짜. 하루 일정이면 null"
    )
    start_time: str | None = None
    end_time: str | None = None
    recurrence: RecurrenceRule | None = Field(
        default=None,
        description="반복하지 않는 일정이면 null"
    )


class AvailableSlot(BaseModel):
    id: str = Field(description="가용 시간대를 구별하기 위한 고유 ID")
    date: str | None = Field(
        default=None,
        description="특정 날짜에만 가능한 경우 YYYY-MM-DD"
    )
    weekdays: list[int] = Field(
        default_factory=list,
        description="반복 가능한 요일. 월요일=0부터 일요일=6까지"
    )
    start_time: str | None = Field(
        default=None,
        description="HH:MM 형식의 시작 시간"
    )
    end_time: str | None = Field(
        default=None,
        description="HH:MM 형식의 종료 시간"
    )
    available_minutes: int | None = Field(
        default=None,
        ge=1,
        description="정확한 시간대 없이 분량만 말한 경우의 가용 시간"
    )


class PlanningContext(BaseModel):
    goal: str
    start_date: str | None = None
    end_date: str | None = None
    tasks: list[Task]
    fixed_events: list[FixedEvent] = Field(default_factory=list)
    available_slots: list[AvailableSlot] = Field(default_factory=list)
    unavailable_weekdays: list[int] = Field(
        default_factory=list,
        description="사용자가 계획할 수 없다고 명시한 반복 요일. 월요일=0"
    )
    planning_strategy: Literal["quick", "balanced", "buffer"] = Field(
        default="quick",
        description="빠른 완료, 균등 분산, 여유일 확보 중 배치 전략"
    )
    missing_information: list[str] = Field(default_factory=list)


class ScheduleItem(BaseModel):
    task_id: str
    title: str
    date: str
    start_time: str
    end_time: str
    minutes: int
    status: Literal["planned", "completed", "missed"] = "planned"


class Plan(BaseModel):
    schedule: list[ScheduleItem]
    warnings: list[str] = Field(default_factory=list)
    explanation: str


class PlanningDecision(BaseModel):
    ordered_task_ids: list[str] = Field(
        description="먼저 배치할 작업부터 나열한 작업 ID"
    )
    explanation: str = Field(
        description="이 순서로 계획한 이유를 사용자가 이해하기 쉽게 설명"
    )


class PlanValidation(BaseModel):
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PlanUpdate(BaseModel):
    completed_task_ids: list[str] = Field(default_factory=list)
    missed_task_ids: list[str] = Field(default_factory=list)
    new_fixed_events: list[FixedEvent] = Field(default_factory=list)
    changed_priorities: dict[str, int] = Field(default_factory=dict)
    additional_notes: str = ""

#추가 질문 생성
class FollowUpQuestion(BaseModel):
    key: str = Field(
        description="질문으로 확인하려는 정보의 식별자"
    )
    question: str = Field(
        description="사용자에게 보여줄 자연스러운 질문"
    )
    example: str | None = Field(
        default=None,
        description="사용자가 참고할 수 있는 짧은 답변 예시"
    )
class FollowUpQuestions(BaseModel):
    questions: list[FollowUpQuestion]
