"""Reference user payments must observe preceding assistant requests."""

import pytest

from tau2.data_model.message import AssistantMessage, Tick, ToolCall, UserMessage
from tau2.data_model.tasks import (
    Action,
    EnvFunctionCall,
    EvaluationCriteria,
    InitialState,
    RewardType,
    Task,
    UserScenario,
)
from tau2.domains.telecom.environment import get_environment
from tau2.evaluator.evaluator_env import (
    EnvironmentEvaluator,
    FullDuplexEnvironmentEvaluator,
)


def payment_task():
    return Task(
        id="payment_request_across_user_and_agent_tools",
        user_scenario=UserScenario(
            instructions="Pay the overdue bill and restore service."
        ),
        initial_state=InitialState(
            initialization_actions=[
                EnvFunctionCall(
                    env_type="user",
                    func_name="set_user_info",
                    arguments={
                        "name": "John Smith",
                        "phone_number": "555-123-2002",
                    },
                ),
                EnvFunctionCall(
                    env_type="assistant",
                    func_name="suspend_line_for_overdue_bill",
                    arguments={
                        "customer_id": "C1001",
                        "line_id": "L1002",
                        "new_bill_id": "B-SYNC-CONTROL",
                        "contract_ended": False,
                    },
                ),
            ]
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=[
                Action(
                    action_id="request",
                    name="send_payment_request",
                    arguments={
                        "customer_id": "C1001",
                        "bill_id": "B-SYNC-CONTROL",
                    },
                ),
                Action(
                    action_id="pay", name="make_payment", requestor="user", arguments={}
                ),
                Action(
                    action_id="resume",
                    name="resume_line",
                    arguments={
                        "customer_id": "C1001",
                        "line_id": "L1002",
                    },
                ),
                Action(
                    action_id="reboot",
                    name="reboot_device",
                    requestor="user",
                    arguments={},
                ),
            ],
            reward_basis=[RewardType.DB],
        ),
    )


@pytest.mark.parametrize("full_duplex", [False, True])
@pytest.mark.parametrize("sync_gold_actions", [False, True])
@pytest.mark.parametrize("control", ["gold", "noop", "valid_wrong"])
def test_payment_state_reference_sync(full_duplex, sync_gold_actions, control):
    task = payment_task()
    environment = get_environment()
    environment.set_state(
        initialization_data=None,
        initialization_actions=task.initial_state.initialization_actions,
        message_history=[],
    )
    messages, ticks = [], []
    actions = [] if control == "noop" else list(task.evaluation_criteria.actions)
    if control == "valid_wrong":
        actions.append(
            Action(
                action_id="extra",
                name="refuel_data",
                arguments={
                    "customer_id": "C1001",
                    "line_id": "L1002",
                    "gb_amount": 1.0,
                },
            )
        )
    for index, action in enumerate(actions):
        call = ToolCall(
            id=action.action_id,
            name=action.name,
            arguments=action.arguments,
            requestor=action.requestor,
        )
        response = environment.get_response(call)
        assert not response.error, response.content
        cls = AssistantMessage if action.requestor == "assistant" else UserMessage
        message = cls(role=action.requestor, content=None, tool_calls=[call])
        messages.extend([message, response])
        tick = dict(tick_id=index, timestamp=f"2025-02-25T12:00:{index:02d}Z")
        if action.requestor == "assistant":
            tick.update(
                agent_chunk=message,
                agent_tool_calls=[call],
                agent_tool_results=[response],
            )
        else:
            tick.update(
                user_chunk=message, user_tool_calls=[call], user_tool_results=[response]
            )
        ticks.append(Tick(**tick))
    evaluator = FullDuplexEnvironmentEvaluator if full_duplex else EnvironmentEvaluator
    verdict = evaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=ticks if full_duplex else messages,
        sync_gold_actions=sync_gold_actions,
    )
    assert verdict.reward == float(control == "gold" and sync_gold_actions)
