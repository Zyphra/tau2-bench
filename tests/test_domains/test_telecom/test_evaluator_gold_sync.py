"""Gold DB comparison must include telecom state derived from a refuel."""

import pytest

from tau2.data_model.message import (
    AssistantMessage,
    Tick,
    ToolCall,
    ToolMessage,
    UserMessage,
)
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


@pytest.fixture(
    params=[
        (EnvironmentEvaluator, False),
        (FullDuplexEnvironmentEvaluator, True),
    ],
    ids=["half_duplex", "full_duplex"],
)
def evaluator_mode(request):
    return request.param


def make_quota_task():
    """Use the existing asset line and its plan to cross the quota boundary."""
    environment = get_environment()
    line = environment.tools._get_line_by_id("L1002")
    plan = environment.tools._get_plan_by_id(line.plan_id)
    assert line.data_refueling_gb == 0
    return Task(
        id="refuel_updates_derived_quota",
        user_scenario=UserScenario(instructions="Add 2 GB to the line."),
        initial_state=InitialState(
            initialization_actions=[
                EnvFunctionCall(
                    env_type="user",
                    func_name="set_user_info",
                    arguments={"name": "John Smith", "phone_number": line.phone_number},
                ),
                EnvFunctionCall(
                    env_type="assistant",
                    func_name="set_data_usage",
                    arguments={
                        "customer_id": "C1001",
                        "line_id": line.line_id,
                        "data_used_gb": plan.data_limit_gb + 1.5,
                    },
                ),
            ]
        ),
        evaluation_criteria=EvaluationCriteria(
            actions=[
                Action(
                    action_id="refuel",
                    name="refuel_data",
                    arguments={
                        "customer_id": "C1001",
                        "line_id": line.line_id,
                        "gb_amount": 2.0,
                    },
                )
            ],
            reward_basis=[RewardType.DB],
        ),
    )


@pytest.fixture
def quota_task():
    return make_quota_task()


def _hashes(environment):
    return environment.get_db_hash(), environment.get_user_db_hash()


def observe_refuel(evaluator_mode, task, amount):
    """Execute a real tool response and evaluate its strictly replayed state."""
    evaluator, full_duplex = evaluator_mode
    environment = get_environment()
    initialization = task.initial_state.initialization_actions
    environment.set_state(
        initialization_data=task.initial_state.initialization_data,
        initialization_actions=initialization,
        message_history=[],
        strict=True,
    )
    assert environment.user_tools.db.surroundings.mobile_data_usage_exceeded
    initial_hashes = _hashes(environment)
    user = UserMessage(role="user", content="Please add 2 GB.")
    messages = [user]
    ticks = [Tick(tick_id=0, timestamp="2025-02-25T12:00:00Z", user_chunk=user)]
    response = None
    if amount is not None:
        call = ToolCall(
            id="refuel",
            name="refuel_data",
            requestor="assistant",
            arguments={"customer_id": "C1001", "line_id": "L1002", "gb_amount": amount},
        )
        if amount < 0:
            with pytest.raises(ValueError, match="Refuel amount must be positive"):
                environment.make_tool_call(call.name, call.requestor, **call.arguments)
            assert _hashes(environment) == initial_hashes
        response = environment.get_response(call)
        assert isinstance(response, ToolMessage)
        assistant = AssistantMessage(role="assistant", tool_calls=[call])
        messages.extend([assistant, response])
        ticks.append(
            Tick(
                tick_id=1,
                timestamp="2025-02-25T12:00:01Z",
                agent_chunk=assistant,
                agent_tool_calls=[call],
                agent_tool_results=[response],
            )
        )
    constructed = []

    def constructor(**kwargs):
        actual = get_environment(**kwargs)
        constructed.append(actual)
        return actual

    reward = evaluator.calculate_reward(
        environment_constructor=constructor,
        task=task,
        full_trajectory=ticks if full_duplex else messages,
        strict_replay=True,
    )
    assert len(constructed) == 2
    predicted, gold = constructed
    assert _hashes(environment) == _hashes(predicted)
    return {
        "reward": reward,
        "response": response,
        "live": environment,
        "predicted": predicted,
        "gold": gold,
        "initial_hashes": initial_hashes,
    }


def assert_refuel_case(observed, amount):
    """Check the end state and preserve unsuccessful controls."""
    reward = observed["reward"]
    response = observed["response"]
    line = observed["live"].tools._get_line_by_id("L1002")
    exceeded = observed["live"].user_tools.db.surroundings.mobile_data_usage_exceeded
    if amount == 2.0:
        assert response.error is False
        assert line.data_refueling_gb == 2.0
        assert exceeded is False
        assert reward.db_check.db_match, "Canonical refuel gold DB mismatch"
        assert _hashes(observed["predicted"]) == _hashes(observed["gold"])
        assert reward.db_check.db_reward == reward.reward == 1.0
    else:
        assert exceeded is True
        assert reward.db_check.db_match is False
        assert reward.db_check.db_reward == reward.reward == 0.0
        if amount == 1.0:
            assert response.error is False
            assert line.data_refueling_gb == 1.0
        elif amount == -1.0:
            assert response.error is True
            assert response.id == "refuel"
            assert response.requestor == "assistant"
            assert "Refuel amount must be positive" in response.content
            assert line.data_refueling_gb == 0.0
            assert _hashes(observed["live"]) == observed["initial_hashes"]
        else:
            assert response is None
            assert line.data_refueling_gb == 0.0
            assert _hashes(observed["live"]) == observed["initial_hashes"]


@pytest.mark.parametrize(
    "amount",
    [2.0, 1.0, -1.0, None],
    ids=["canonical", "wrong_amount", "typed_error", "initial_only"],
)
def test_refuel_gold_state(evaluator_mode, quota_task, amount):
    assert_refuel_case(observe_refuel(evaluator_mode, quota_task, amount), amount)
