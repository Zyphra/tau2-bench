"""Tests for rescoring trajectories with tau2.scripts.evaluate_trajectories.

Regression coverage for full-duplex (voice) results: rescoring must detect
the communication mode from the results and use the tick-based evaluators,
instead of silently evaluating simulation.messages with the half-duplex
evaluators (reported in PR #386).
"""

import asyncio
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from nemo_gym.global_config import (
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    get_global_config_dict,
)
from nemo_gym.server_utils import (
    GlobalAIOHTTPAsyncClientConfig,
    close_global_aiohttp_client,
    get_global_aiohttp_client,
    is_global_aiohttp_client_setup,
    set_global_aiohttp_client,
)

from tau2.cli import main as cli_main
from tau2.config import DEFAULT_LLM_NL_ASSERTIONS_ARGS
from tau2.data_model.message import AssistantMessage, Tick, ToolCall, ToolMessage
from tau2.data_model.simulation import (
    AudioNativeConfig,
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import EvaluationCriteria, RewardType, Task, UserScenario
from tau2.environment.environment import EnvironmentInfo
from tau2.orchestrator.modes import CommunicationMode
from tau2.registry import registry
from tau2.run import get_tasks
from tau2.scripts import evaluate_trajectories as evaluate_trajectories_module
from tau2.scripts.evaluate_trajectories import (
    _build_eval_env_kwargs,
    compute_simulation_rewards,
    compute_simulation_rewards_async,
    evaluate_trajectories,
    get_communication_mode,
)

# ---- Fixtures ----


def _make_info(
    user_implementation: str = "user_simulator",
    audio_native_config: AudioNativeConfig = None,
) -> Info:
    return Info(
        git_commit="abc123",
        num_trials=1,
        max_steps=100,
        max_errors=10,
        user_info=UserInfo(implementation=user_implementation),
        agent_info={"implementation": "llm_agent"},
        environment_info=EnvironmentInfo(domain_name="mock", policy="test policy"),
        audio_native_config=audio_native_config,
    )


def _make_task(task_id: str) -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(instructions="test instruction"),
        evaluation_criteria=EvaluationCriteria(),
    )


def _make_half_duplex_sim(task_id: str) -> SimulationRun:
    return SimulationRun(
        id=f"sim-{task_id}",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=[],
    )


def _make_full_duplex_sim(task_id: str, ticks: list[Tick] = None) -> SimulationRun:
    """Full-duplex sims store ticks and no messages (see FullDuplexOrchestrator)."""
    return SimulationRun(
        id=f"sim-{task_id}",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=TerminationReason.USER_STOP,
        messages=None,
        ticks=ticks if ticks is not None else [],
        mode=CommunicationMode.FULL_DUPLEX.value,
    )


# ---- Mode detection ----


class TestGetCommunicationMode:
    def test_half_duplex_by_default(self):
        sim = _make_half_duplex_sim("t0")
        results = Results(
            info=_make_info(),
            tasks=[_make_task("t0")],
            simulations=[sim],
        )
        assert get_communication_mode(results, sim) == CommunicationMode.HALF_DUPLEX

    def test_detects_voice_streaming_user_implementation(self):
        results = Results(
            info=_make_info(user_implementation="voice_streaming_user_simulator"),
            tasks=[_make_task("t0")],
            simulations=[],
        )
        assert get_communication_mode(results) == CommunicationMode.FULL_DUPLEX

    def test_detects_audio_native_config(self):
        results = Results(
            info=_make_info(audio_native_config=AudioNativeConfig()),
            tasks=[_make_task("t0")],
            simulations=[],
        )
        assert get_communication_mode(results) == CommunicationMode.FULL_DUPLEX

    def test_detects_full_duplex_simulation_mode(self):
        # Isolate the mode-field signal: no ticks stored (e.g. saved without
        # verbose tick data) and no run-level voice info.
        sim = _make_full_duplex_sim("t0")
        sim.ticks = None
        results = Results(
            info=_make_info(),
            tasks=[_make_task("t0")],
            simulations=[sim],
        )
        assert get_communication_mode(results, sim) == CommunicationMode.FULL_DUPLEX

    def test_detects_ticks_when_mode_missing(self):
        # Older full-duplex trajectories may predate the SimulationRun.mode
        # field and deserialize with the half_duplex default.
        sim = _make_full_duplex_sim("t0")
        sim.mode = CommunicationMode.HALF_DUPLEX.value
        results = Results(
            info=_make_info(),
            tasks=[_make_task("t0")],
            simulations=[sim],
        )
        assert get_communication_mode(results, sim) == CommunicationMode.FULL_DUPLEX

    def test_mixed_results_detected_per_simulation(self):
        full_duplex_sim = _make_full_duplex_sim("t0")
        half_duplex_sim = _make_half_duplex_sim("t1")
        results = Results(
            info=_make_info(),
            tasks=[_make_task("t0"), _make_task("t1")],
            simulations=[full_duplex_sim, half_duplex_sim],
        )
        assert (
            get_communication_mode(results, full_duplex_sim)
            == CommunicationMode.FULL_DUPLEX
        )
        assert (
            get_communication_mode(results, half_duplex_sim)
            == CommunicationMode.HALF_DUPLEX
        )


# ---- Rescoring passes the detected mode to the evaluator ----


class TestComputeSimulationRewardsMode:
    def _capture_mode(self, monkeypatch, results) -> list:
        captured = []

        original_evaluator = evaluate_trajectories_module.evaluate_simulation

        async def recording_evaluator(**kwargs):
            captured.append(kwargs)
            return await original_evaluator(**kwargs)

        monkeypatch.setattr(
            evaluate_trajectories_module,
            "evaluate_simulation",
            recording_evaluator,
        )
        compute_simulation_rewards(results)
        return captured

    def test_full_duplex_results_evaluated_in_full_duplex_mode(self, monkeypatch):
        results = Results(
            info=_make_info(user_implementation="voice_streaming_user_simulator"),
            tasks=[_make_task("t0")],
            simulations=[_make_full_duplex_sim("t0")],
        )
        captured = self._capture_mode(monkeypatch, results)
        assert len(captured) == 1
        assert captured[0]["mode"] == CommunicationMode.FULL_DUPLEX

    def test_half_duplex_results_evaluated_in_half_duplex_mode(self, monkeypatch):
        results = Results(
            info=_make_info(),
            tasks=[_make_task("t0")],
            simulations=[_make_half_duplex_sim("t0")],
        )
        captured = self._capture_mode(monkeypatch, results)
        assert len(captured) == 1
        assert captured[0]["mode"] == CommunicationMode.HALF_DUPLEX


# ---- End-to-end: rescoring a full-duplex results file uses tick evaluators ----


class TestFullDuplexRescoring:
    def test_rescoring_full_duplex_results_uses_ticks(self):
        """Regression test for PR #386: rescoring full-duplex results must
        evaluate simulation.ticks. The golden create_task action below only
        exists in the ticks (messages is None, as in real voice trajectories),
        so the half-duplex evaluators cannot produce this reward."""
        task = get_tasks("mock", task_ids=["create_task_1"])[0]
        tick = Tick(
            tick_id=0,
            timestamp="2026-01-01T00:00:30",
            agent_tool_calls=[
                ToolCall(
                    id="call_1",
                    name="create_task",
                    arguments={"user_id": "user_1", "title": "Important Meeting"},
                )
            ],
            agent_tool_results=[
                ToolMessage(
                    id="call_1",
                    role="tool",
                    content='{"task_id": "task_2", "title": "Important Meeting", '
                    '"description": null, "status": "pending"}',
                    requestor="assistant",
                )
            ],
        )
        results = Results(
            info=_make_info(user_implementation="voice_streaming_user_simulator"),
            tasks=[task],
            simulations=[_make_full_duplex_sim(task.id, ticks=[tick])],
        )

        rescored = compute_simulation_rewards(results)

        reward_info = rescored.simulations[0].reward_info
        assert reward_info.reward == 1.0
        assert reward_info.db_check is not None
        assert reward_info.db_check.db_match is True
        # The tick-based action evaluator matched the create_task call.
        assert reward_info.action_checks is not None
        assert all(check.action_match for check in reward_info.action_checks)


# ---- Re-grading options: strict_replay, env_kwargs, fresh tasks ----


class TestRegradingOptions:
    def _capture_eval_kwargs(self, monkeypatch, results, **compute_kwargs):
        captured = []

        original_evaluator = evaluate_trajectories_module.evaluate_simulation

        async def recording_evaluator(**kwargs):
            captured.append(kwargs)
            return await original_evaluator(**kwargs)

        monkeypatch.setattr(
            evaluate_trajectories_module,
            "evaluate_simulation",
            recording_evaluator,
        )
        compute_simulation_rewards(results, **compute_kwargs)
        return captured

    def test_rescoring_uses_lenient_replay(self, monkeypatch):
        """Re-grading replays historical trajectories whose recorded tool
        outputs may cosmetically predate current tool code; the replay must
        not abort on output-text drift."""
        results = Results(
            info=_make_info(),
            tasks=[_make_task("t0")],
            simulations=[_make_half_duplex_sim("t0")],
        )
        captured = self._capture_eval_kwargs(monkeypatch, results)
        assert captured[0]["strict_replay"] is False

    def test_banking_read_log_allowlist_derivation(self):
        """banking_knowledge live grading logs golden-trajectory reads to the
        agent_discoverable_tools table via a per-task allowlist; re-grading
        must pass the same allowlist or required-read assertions silently
        stop discriminating."""
        from tau2.data_model.tasks import Action

        task = _make_task("t0")
        task.evaluation_criteria = EvaluationCriteria(
            actions=[
                Action(
                    action_id="t0_0",
                    requestor="assistant",
                    name="call_discoverable_agent_tool",
                    arguments={
                        "agent_tool_name": "get_bank_account_transactions_9173",
                        "arguments": "{}",
                    },
                )
            ]
        )
        # Exercise the real derivation without constructing the domain's
        # default retrieval environment, which can require network services.
        assert _build_eval_env_kwargs("banking_knowledge", task) == {
            "read_log_allowlist": {"get_bank_account_transactions_9173"}
        }

    def test_non_banking_domain_gets_no_env_kwargs(self, monkeypatch):
        results = Results(
            info=_make_info(),
            tasks=[_make_task("t0")],
            simulations=[_make_half_duplex_sim("t0")],
        )
        captured = self._capture_eval_kwargs(monkeypatch, results)
        assert captured[0]["env_kwargs"] is None

    def test_fresh_tasks_reloads_task_definitions(self, monkeypatch):
        """--fresh-tasks must grade against the current data-dir task
        definitions, not the ones embedded in the results file."""
        embedded_task = get_tasks("mock", task_ids=["create_task_1"])[0]
        embedded_task = embedded_task.model_copy(deep=True)
        embedded_task.evaluation_criteria = EvaluationCriteria()  # stale criteria
        results = Results(
            info=_make_info(),
            tasks=[embedded_task],
            simulations=[_make_half_duplex_sim(embedded_task.id)],
        )

        captured = self._capture_eval_kwargs(monkeypatch, results, fresh_tasks=True)
        current_task = get_tasks("mock", task_ids=["create_task_1"])[0]
        assert (
            captured[0]["task"].evaluation_criteria == current_task.evaluation_criteria
        )
        assert captured[0]["task"].evaluation_criteria != EvaluationCriteria()

    def test_embedded_tasks_used_by_default(self, monkeypatch):
        embedded_task = get_tasks("mock", task_ids=["create_task_1"])[0]
        embedded_task = embedded_task.model_copy(deep=True)
        embedded_task.evaluation_criteria = EvaluationCriteria()
        results = Results(
            info=_make_info(),
            tasks=[embedded_task],
            simulations=[_make_half_duplex_sim(embedded_task.id)],
        )
        captured = self._capture_eval_kwargs(monkeypatch, results)
        assert captured[0]["task"].evaluation_criteria == EvaluationCriteria()


# ---- Sync/async boundary and concrete Results behavior ----


def _make_mixed_results() -> Results:
    """Real mock-domain tool output in both trajectory representations."""
    task = get_tasks("mock", task_ids=["create_task_1"])[0]
    call = ToolCall(
        id="create_1",
        name="create_task",
        arguments={"user_id": "user_1", "title": "Important Meeting"},
    )
    response = registry.get_env_constructor("mock")().get_response(call)
    assert not response.error
    half = _make_half_duplex_sim(task.id)
    half.id = "half-success"
    half.messages = [AssistantMessage(role="assistant", tool_calls=[call]), response]
    full = _make_full_duplex_sim(
        task.id,
        ticks=[
            Tick(
                tick_id=0,
                timestamp="2026-01-01T00:00:30",
                agent_tool_calls=[call.model_copy(deep=True)],
                agent_tool_results=[response.model_copy(deep=True)],
            )
        ],
    )
    full.id = "full-success"
    missing = _make_half_duplex_sim(task.id)
    missing.id = "half-missing-action"
    info = _make_info()
    info.num_trials = 3
    return Results(info=info, tasks=[task], simulations=[half, full, missing])


def _assert_mixed_rewards(results: Results) -> None:
    assert [sim.id for sim in results.simulations] == [
        "half-success",
        "full-success",
        "half-missing-action",
    ]
    assert all(isinstance(sim.reward_info, RewardInfo) for sim in results.simulations)
    assert [sim.reward_info.reward for sim in results.simulations] == [1.0, 1.0, 0.0]
    assert [sim.reward_info.db_check.db_match for sim in results.simulations] == [
        True,
        True,
        False,
    ]


class TestAsyncRegrading:
    def test_mixed_trajectories_are_graded_without_mutating_input(self):
        results = _make_mixed_results()
        for sim in results.simulations:
            sim.reward_info = RewardInfo(reward=0.25)
        original = results.model_dump(mode="json")

        rescored = compute_simulation_rewards(results)

        _assert_mixed_rewards(rescored)
        assert results.model_dump(mode="json") == original
        assert rescored is not results
        assert rescored.tasks[0] is not results.tasks[0]
        assert (
            rescored.simulations[0].messages[0]
            is not results.simulations[0].messages[0]
        )
        rescored.tasks[0].user_scenario.instructions = "changed output"
        rescored.simulations[0].messages[0].tool_calls[0].arguments["title"] = "changed"
        rescored.simulations[0].reward_info.reward = 0.5
        assert results.model_dump(mode="json") == original

    def test_sync_and_async_helpers_agree(self):
        results = _make_mixed_results()
        original = results.model_dump(mode="json")
        synchronous = compute_simulation_rewards(results)

        async def grade_in_running_loop():
            assert asyncio.get_running_loop().is_running()
            return await compute_simulation_rewards_async(results)

        asynchronous = asyncio.run(grade_in_running_loop())

        _assert_mixed_rewards(synchronous)
        _assert_mixed_rewards(asynchronous)
        assert asynchronous.model_dump(mode="json") == synchronous.model_dump(
            mode="json"
        )
        assert results.model_dump(mode="json") == original

    def test_sync_helper_refuses_an_active_loop(self):
        results = _make_mixed_results()
        original = results.model_dump(mode="json")

        async def call_sync_helper():
            with pytest.raises(
                RuntimeError, match="await compute_simulation_rewards_async"
            ):
                compute_simulation_rewards(results)
            # Refusal must leave the running loop usable for real evaluation.
            return await compute_simulation_rewards_async(results)

        _assert_mixed_rewards(asyncio.run(call_sync_helper()))
        assert results.model_dump(mode="json") == original

    @pytest.mark.parametrize("fresh_tasks", [False, True])
    def test_task_refresh_preserves_original_results(self, fresh_tasks):
        results = _make_mixed_results()
        results.tasks[0].evaluation_criteria = EvaluationCriteria()
        original = results.model_dump(mode="json")

        rescored = compute_simulation_rewards(results, fresh_tasks=fresh_tasks)

        if fresh_tasks:
            _assert_mixed_rewards(rescored)
        else:
            assert [sim.reward_info.reward for sim in rescored.simulations] == [1.0] * 3
        assert results.model_dump(mode="json") == original
        assert results.tasks[0].evaluation_criteria == EvaluationCriteria()

    @pytest.mark.parametrize("full_duplex", [False, True])
    def test_empty_nl_assertions_use_real_async_evaluator(self, full_duplex):
        task = _make_task("empty-nl")
        task.evaluation_criteria = EvaluationCriteria(
            nl_assertions=[], reward_basis=[RewardType.NL_ASSERTION]
        )
        sim = (
            _make_full_duplex_sim(task.id)
            if full_duplex
            else _make_half_duplex_sim(task.id)
        )
        results = Results(info=_make_info(), tasks=[task], simulations=[sim])

        rescored = compute_simulation_rewards(results)

        reward = rescored.simulations[0].reward_info
        assert isinstance(reward, RewardInfo)
        assert reward.reward == 1.0
        assert reward.nl_assertions == []
        assert reward.reward_breakdown == {RewardType.NL_ASSERTION: 1.0}
        assert results.simulations[0].reward_info is None


class TestRegradingFiles:
    def test_sync_file_entry_point_roundtrips_rewards(self, tmp_path):
        source = tmp_path / "mixed.json"
        output = tmp_path / "graded"
        original = _make_mixed_results()
        original.save(source)

        evaluate_trajectories([str(source)], output_dir=str(output))

        _assert_mixed_rewards(Results.load(output / "updated_mixed.json"))
        assert Results.load(source).model_dump(mode="json") == original.model_dump(
            mode="json"
        )

    def test_bad_file_does_not_prevent_good_file_from_being_saved(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("invalid json")
        good = tmp_path / "good.json"
        _make_mixed_results().save(good)
        output = tmp_path / "graded"

        with pytest.raises(SystemExit) as raised:
            evaluate_trajectories([str(bad), str(good)], output_dir=str(output))

        assert raised.value.code == 1
        assert not (output / "updated_bad.json").exists()
        _assert_mixed_rewards(Results.load(output / "updated_good.json"))


# ---- Real NL evaluator / NeMo HTTP client lifecycle ----


@pytest.fixture
def local_nl_provider(monkeypatch):
    """Finite loopback HTTP fixture; real evaluator and client, no model service."""
    assert not is_global_aiohttp_client_setup()
    monkeypatch.setenv(
        NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
        json.dumps(
            {
                "global_aiohttp_connector_limit": 4,
                "global_aiohttp_connector_limit_per_host": 4,
            }
        ),
    )
    # This is NeMo's public parent-injected configuration interface.
    get_global_config_dict()
    state = {"requests": [], "clients": [], "invalid_content": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, request))
            state["clients"].append(get_global_aiohttp_client())
            # Bound every test's fixture to four requests, including mistakes.
            if len(state["requests"]) > 4:
                self.send_error(429, "local fixture request limit")
                return
            content = (
                "invalid json"
                if state["invalid_content"]
                else json.dumps(
                    {
                        "results": [
                            {
                                "expectedOutcome": "local assertion",
                                "metExpectation": True,
                                "reasoning": "finite local fixture response",
                            }
                        ]
                    }
                )
            )
            response = json.dumps(
                {
                    "id": "local-completion",
                    "object": "chat.completion",
                    "created": 0,
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    monkeypatch.setitem(
        DEFAULT_LLM_NL_ASSERTIONS_ARGS,
        "api_base",
        f"http://127.0.0.1:{server.server_port}/v1",
    )
    monkeypatch.setitem(DEFAULT_LLM_NL_ASSERTIONS_ARGS, "api_key", "")
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not is_global_aiohttp_client_setup()


def _make_nl_results(full_duplex=False) -> Results:
    task = _make_task("nonempty-nl")
    task.evaluation_criteria = EvaluationCriteria(
        nl_assertions=["local assertion"], reward_basis=[RewardType.NL_ASSERTION]
    )
    sim = (
        _make_full_duplex_sim(task.id)
        if full_duplex
        else _make_half_duplex_sim(task.id)
    )
    return Results(info=_make_info(), tasks=[task], simulations=[sim])


def _assert_real_nl_reward(results: Results):
    reward = results.simulations[0].reward_info
    assert isinstance(reward, RewardInfo)
    assert reward.reward == 1.0
    assert len(reward.nl_assertions) == 1
    assert reward.nl_assertions[0].nl_assertion == "local assertion"
    assert reward.nl_assertions[0].met is True


class TestRegradingClientLifecycle:
    @pytest.mark.parametrize("full_duplex", [False, True])
    def test_sequential_sync_calls_close_their_real_http_clients(
        self, local_nl_provider, full_duplex
    ):
        results = _make_nl_results(full_duplex)
        for _ in range(2):
            _assert_real_nl_reward(compute_simulation_rewards(results))
            assert not is_global_aiohttp_client_setup()
            assert local_nl_provider["clients"][-1].closed
        assert len(local_nl_provider["requests"]) == 2
        assert local_nl_provider["clients"][0] is not local_nl_provider["clients"][1]
        assert all(
            path == "/v1/chat/completions" for path, _ in local_nl_provider["requests"]
        )
        assert results.simulations[0].reward_info is None

    def test_multiple_files_use_fresh_owned_clients(
        self, local_nl_provider, tmp_path, monkeypatch
    ):
        paths = [tmp_path / "half.json", tmp_path / "full.json"]
        for path, full in zip(paths, [False, True], strict=True):
            _make_nl_results(full).save(path)
        output = tmp_path / "graded"

        monkeypatch.setattr(
            sys,
            "argv",
            ["tau2", "evaluate-trajs", *map(str, paths), "--output-dir", str(output)],
        )
        cli_main()

        for path in paths:
            _assert_real_nl_reward(Results.load(output / f"updated_{path.name}"))
        assert len(local_nl_provider["requests"]) == 2
        assert not is_global_aiohttp_client_setup()
        assert all(client.closed for client in local_nl_provider["clients"])
        assert local_nl_provider["clients"][0] is not local_nl_provider["clients"][1]

    def test_sync_refuses_and_preserves_an_existing_foreign_client(
        self, local_nl_provider
    ):
        loop = asyncio.new_event_loop()

        async def create_client():
            return set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())

        client = loop.run_until_complete(create_client())
        try:
            with pytest.raises(RuntimeError, match="existing NeMo HTTP client.*await"):
                compute_simulation_rewards(_make_nl_results())
            assert not loop.is_closed()
            assert not client.closed
            assert get_global_aiohttp_client() is client
            assert local_nl_provider["requests"] == []
            # It remains usable on its actual owning loop after sync refusal.
            _assert_real_nl_reward(
                loop.run_until_complete(
                    compute_simulation_rewards_async(_make_nl_results())
                )
            )
            assert not client.closed
            assert get_global_aiohttp_client() is client
        finally:
            loop.run_until_complete(close_global_aiohttp_client())
            loop.close()
        assert client.closed

    @pytest.mark.parametrize("existing_client", [False, True])
    def test_async_helper_keeps_client_lifecycle_with_the_caller(
        self, local_nl_provider, existing_client
    ):
        async def scenario():
            client = (
                set_global_aiohttp_client(GlobalAIOHTTPAsyncClientConfig())
                if existing_client
                else None
            )
            try:
                for _ in range(2):
                    _assert_real_nl_reward(
                        await compute_simulation_rewards_async(_make_nl_results())
                    )
                    observed = get_global_aiohttp_client()
                    if client is None:
                        client = observed
                    assert observed is client
                    assert not client.closed
                assert local_nl_provider["clients"] == [client, client]
            finally:
                await close_global_aiohttp_client()
            assert client.closed

        asyncio.run(scenario())
        assert len(local_nl_provider["requests"]) == 2

    def test_sync_closes_owned_client_when_nl_response_is_invalid(
        self, local_nl_provider
    ):
        local_nl_provider["invalid_content"] = True

        with pytest.raises(json.JSONDecodeError):
            compute_simulation_rewards(_make_nl_results())

        assert len(local_nl_provider["requests"]) == 1
        assert local_nl_provider["clients"][0].closed
        assert not is_global_aiohttp_client_setup()
