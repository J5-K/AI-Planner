from dotenv import load_dotenv
from datetime import date, timedelta
import re
from langchain_openai import ChatOpenAI

from models import FixedEvent, FollowUpQuestion, PlanningContext, FollowUpQuestions
from planner import (
    apply_suggested_estimates,
    apply_text_constraints,
    build_question_needs,
    detect_missing_information,
    create_plan,
    replan,
)
from scheduler import expand_fixed_events
from prompts import extract_prompt, question_prompt



load_dotenv()

model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

extract_model = model.with_structured_output(PlanningContext, method="function_calling",)

extract_chain = extract_prompt | extract_model

question_model = model.with_structured_output(
    FollowUpQuestions,
    method="function_calling",
)

question_chain = question_prompt | question_model

MAX_QUESTION_ROUNDS = 2
MAX_QUESTIONS_PER_ROUND = 4


def fallback_question(need):
    """질문 생성 LLM이 필수 항목을 빠뜨렸을 때 사용할 안전한 질문."""
    templates = {
        "plan_period": (
            "이 계획은 언제부터 언제까지 진행할까요?",
            "오늘부터 한 달간",
        ),
        "tasks": (
            "각 목표를 위해 실제로 어떤 활동이나 작업을 할 예정인가요?",
            "주 3회 운동, 주제 선정과 핵심 기능 개발",
        ),
        "availability": (
            "평일과 주말에는 각각 언제 계획을 실행할 수 있나요?",
            "평일 저녁 9시~11시, 토요일 낮 12시~6시",
        ),
        "availability_detail": (
            "말씀해 주신 가용 시간의 시작과 종료 시각을 알려주시겠어요?",
            "평일은 밤 9시부터 11시까지",
        ),
        "fixed_event_details": (
            "고정 일정은 언제, 몇 시부터 몇 시까지인가요?",
            "매주 수요일 오후 7시부터 9시까지",
        ),
        "task_frequency": (
            "반복 작업은 일주일에 몇 번 하고 싶으신가요?",
            "운동은 주 3회",
        ),
    }
    question, example = templates.get(
        need.key,
        (f"{need.topic}에 대해 조금 더 알려주시겠어요?", None),
    )
    return FollowUpQuestion(key=need.key, question=question, example=example)

def read_multiline_answer() -> str:
    print("\n질문 번호에 맞춰 답해주세요.")
    print("입력이 끝나면 빈 줄에서 Enter를 누르세요.\n")

    lines = []

    while True:
        line = input()

        if not line.strip():
            break

        lines.append(line)

    return "\n".join(lines)


def pair_questions_and_answers(
    question_texts: list[str],
    answers: str,
) -> str:
    """입력 순서대로 각 질문과 답변을 명시적으로 연결한다."""
    answer_lines = [
        line.strip()
        for line in answers.splitlines()
        if line.strip()
    ]
    pairs = []
    for index, question in enumerate(question_texts):
        answer = (
            answer_lines[index]
            if index < len(answer_lines)
            else "답변 없음"
        )
        pairs.append(f"질문: {question}\n답변: {answer}")
    return "\n\n".join(pairs)

def complete_planning_context(original_input: str) -> PlanningContext:
    """최초 요청을 분석하고, 필요한 경우 답변을 받아 계획 조건을 보완한다."""
    current_date = date.today().isoformat()
    context = extract_chain.invoke({
        "current_date": current_date,
        "user_input": original_input,
    })
    context = apply_text_constraints(context, original_input)
    context.missing_information = detect_missing_information(context)

    print("구조화 결과")
    print(context.model_dump())

    conversation_parts = [f"[최초 요청]\n{original_input}"]
    asked_keys: set[str] = set()

    for round_number in range(1, MAX_QUESTION_ROUNDS + 1):
        question_needs = build_question_needs(context)
        if not question_needs:
            completed = apply_suggested_estimates(context)
            completed.missing_information = []
            print("\n계획 초안 생성에 필요한 정보가 준비됐습니다.")
            print_suggested_estimates(completed)
            return completed

        selected_needs = question_needs[:MAX_QUESTIONS_PER_ROUND]
        questions = question_chain.invoke({
            "context": context.model_dump_json(indent=2),
            "question_needs": [need.to_dict() for need in selected_needs],
            "round_number": round_number,
            "asked_keys": sorted(asked_keys),
        })

        selected_keys = {need.key for need in selected_needs}
        questions.questions = [
            item for item in questions.questions
            if item.key in selected_keys
        ]
        returned_keys = {item.key for item in questions.questions}
        questions.questions.extend(
            fallback_question(need)
            for need in selected_needs
            if need.key not in returned_keys
        )

        print(f"\n추가 질문 {round_number}차")
        print("정확하지 않아도 괜찮으니 현재 생각한 범위에서 답해주세요.\n")

        question_texts = []
        for index, item in enumerate(questions.questions, start=1):
            question_text = f"{index}. {item.question}"
            question_texts.append(question_text)
            print(question_text)

            if item.example:
                clean_example = re.sub(r"^예\s*:\s*", "", item.example.strip())
                print(f"   예: {clean_example}")

        answers = read_multiline_answer()
        paired_answers = pair_questions_and_answers(question_texts, answers)
        conversation_parts.append(
            f"[질문과 사용자 답변 - {round_number}차]\n{paired_answers}"
        )
        asked_keys.update(need.key for need in selected_needs)

        combined_input = f"""
{chr(10).join(conversation_parts)}

[현재까지 구조화된 정보]
{context.model_dump_json(indent=2)}

모든 정보를 종합해 PlanningContext를 다시 작성한다.

병합 규칙:
- 각 질문과 바로 아래 답변을 하나의 쌍으로 해석한다.
- 답변 순서를 다른 질문에 연결하지 않는다.
- 사용자의 가장 최근 답변을 우선한다.
- 최근 답변과 충돌하지 않는 기존 정보는 유지한다.
- 사용자가 답하지 않은 정보는 임의로 만들지 않는다.
- 자연어 시간은 분 단위 정수로 변환한다.
- 목표와 실행 가능한 작업을 구분한다.
- 반복 작업과 고정 반복 일정을 구분한다.
- 기존 고정 일정의 반복 규칙을 유지하고 새 답변의 시간을 반영한다.
- 1시간 반은 90분, 2시간 반은 150분으로 변환한다.
- '하루', '반나절'처럼 실제 작업시간이 불분명한 표현은 분으로 확정하지 않는다.
- 최신 답변에 완전한 시작·종료 시간이 있으면 기존 시간 범위를 모두 교체한다.
"""

        context = extract_chain.invoke({
            "current_date": current_date,
            "user_input": combined_input,
        })
        context = apply_text_constraints(context, combined_input)

        print("\n보완된 구조화 결과")
        context.missing_information = detect_missing_information(context)
        print(context.model_dump())

    remaining_needs = build_question_needs(context)
    if remaining_needs:
        print("\n계획 초안을 만들기 위해 다음 필수 정보가 필요합니다.")
        for need in remaining_needs:
            print(f"- {need.topic}")
        return context

    completed = apply_suggested_estimates(context)
    completed.missing_information = []
    print("\n계획 초안 생성에 필요한 정보가 준비됐습니다.")
    print_suggested_estimates(completed)
    return completed


def print_suggested_estimates(context: PlanningContext) -> None:
    suggested_tasks = [
        task for task in context.tasks
        if task.estimate_source == "suggested"
    ]
    if not suggested_tasks:
        return

    print("\n다음 작업 시간은 초안 생성을 위한 임시 제안입니다.")
    print("계획을 확인한 뒤 언제든 수정할 수 있습니다.")
    for task in suggested_tasks:
        print(f"- {task.title}: {task.estimated_minutes}분 (AI 임시 제안)")


def print_expanded_fixed_events(context: PlanningContext) -> None:
    """계획 기간과 고정 일정 정보가 준비된 경우 실제 날짜를 출력한다."""
    remaining_missing = detect_missing_information(context)
    if remaining_missing or not context.start_date or not context.end_date:
        return

    events = expand_fixed_events(
        context.fixed_events,
        context.start_date,
        context.end_date,
    )

    if not events:
        return

    print("\n계획 기간의 고정 일정")
    for event in events:
        print(
            f"- {event.date} "
            f"{event.start_time}~{event.end_time} "
            f"{event.title}"
        )


def print_plan(title, plan, validation) -> None:
    print(f"\n{title}")
    for item in plan.schedule:
        print(f"- {item.date} {item.start_time}~{item.end_time} {item.title}")
    for warning in plan.warnings:
        print(f"경고: {warning}")
    print(f"설명: {plan.explanation}")
    print(f"검증 결과: {'통과' if validation.is_valid else '실패'}")
    for error in validation.errors:
        print(f"- {error}")


def main() -> None:
    original_input = """
한 달 안에 5kg을 감량하고 싶고
AI 개인 미니 프로젝트도 진행하고 싶어.
매주 수요일 저녁에는 약속이 있어.
"""

    completed_context = complete_planning_context(original_input)
    print_expanded_fixed_events(completed_context)
    if build_question_needs(completed_context):
        return

    plan, validation = create_plan(completed_context, model)
    print_plan("최초 계획", plan, validation)

    # 발표 시연용: 첫 일정 실패와 같은 날의 새 약속을 반영한다.
    if plan.schedule:
        missed_id = plan.schedule[0].task_id
        new_event_date = (
            date.fromisoformat(plan.schedule[0].date) + timedelta(days=1)
        ).isoformat()
        new_event = FixedEvent(
            id="new-event", title="새 약속", date=new_event_date,
            start_time="20:00", end_time="21:00",
        )
        updated_plan, updated_validation = replan(
            completed_context, plan, [missed_id], [new_event], model
        )
        print_plan("재계획", updated_plan, updated_validation)


if __name__ == "__main__":
    main()
