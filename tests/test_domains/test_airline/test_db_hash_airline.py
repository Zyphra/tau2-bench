from itertools import permutations

import pytest

from tau2.data_model.message import AssistantMessage, Tick, ToolCall
from tau2.data_model.tasks import Task
from tau2.domains.airline.data_model import FlightDB, Passenger
from tau2.domains.airline.tools import AirlineTools
from tau2.domains.mock.data_model import MockDB
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.telecom.data_model import TelecomDB
from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolKitBase
from tau2.evaluator.evaluator_env import (
    EnvironmentEvaluator,
    FullDuplexEnvironmentEvaluator,
)
from tau2.utils import get_pydantic_hash


@pytest.fixture
def db():
    return FlightDB(
        flights={},
        users={},
        reservations={
            "BOOK01": {
                "reservation_id": "BOOK01",
                "user_id": "customer",
                "origin": "PHL",
                "destination": "LAX",
                "flight_type": "one_way",
                "cabin": "economy",
                "flights": [
                    {
                        "flight_number": "HAT001",
                        "origin": "PHL",
                        "destination": "LGA",
                        "date": "2024-05-16",
                        "price": 100,
                    },
                    {
                        "flight_number": "HAT002",
                        "origin": "LGA",
                        "destination": "LAX",
                        "date": "2024-05-16",
                        "price": 200,
                    },
                ],
                "passengers": [
                    {"first_name": "Kenneth", "last_name": "Li", "dob": "1960-01-01"},
                    {"first_name": "Caleb", "last_name": "Li", "dob": "1990-02-02"},
                    {"first_name": "Eric", "last_name": "Li", "dob": "1992-03-03"},
                ],
                "payment_history": [
                    {"payment_id": "card", "amount": 500},
                    {"payment_id": "card", "amount": -200},
                ],
                "created_at": "2024-05-15T12:00:00",
                "total_baggages": 3,
                "nonfree_baggages": 0,
                "insurance": "no",
            }
        },
    )


@pytest.mark.parametrize("order", list(permutations(range(3))))
@pytest.mark.parametrize("duplicates", [False, True])
def test_passenger_permutations_preserve_hash_without_mutation(db, order, duplicates):
    if duplicates:
        db.reservations["BOOK01"].passengers[2] = (
            db.reservations["BOOK01"].passengers[0].model_copy(deep=True)
        )
    alternate = db.model_copy(deep=True)
    passengers = alternate.reservations["BOOK01"].passengers
    alternate.reservations["BOOK01"].passengers = [passengers[i] for i in order]
    before = alternate.model_dump()
    assert db.get_hash() == alternate.get_hash()
    assert AirlineTools(db).get_db_hash() == AirlineTools(alternate).get_db_hash()
    assert alternate.model_dump() == before


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_single_passengers(db, count):
    db.reservations["BOOK01"].passengers = db.reservations["BOOK01"].passengers[:count]
    before = db.model_dump()
    assert db.get_hash() == db.model_copy(deep=True).get_hash()
    assert db.model_dump() == before


@pytest.mark.parametrize("field", list(Passenger.model_fields))
def test_every_passenger_field_matters(db, field):
    alternate = db.model_copy(deep=True)
    setattr(alternate.reservations["BOOK01"].passengers[0], field, "changed")
    assert db.get_hash() != alternate.get_hash()
    assert AirlineTools(db).get_db_hash() != AirlineTools(alternate).get_db_hash()


@pytest.mark.parametrize("change", ["remove", "add", "substitute"])
def test_passenger_multiplicity_matters(db, change):
    alternate = db.model_copy(deep=True)
    passengers = alternate.reservations["BOOK01"].passengers
    if change == "remove":
        passengers.pop()
    elif change == "add":
        passengers.append(passengers[0].model_copy(deep=True))
    else:
        passengers[2] = passengers[0].model_copy(deep=True)
    assert db.get_hash() != alternate.get_hash()
    assert AirlineTools(db).get_db_hash() != AirlineTools(alternate).get_db_hash()


@pytest.mark.parametrize("field", ["flights", "payment_history"])
def test_ordered_reservation_fields_remain_ordered(db, field):
    alternate = db.model_copy(deep=True)
    getattr(alternate.reservations["BOOK01"], field).reverse()
    assert db.get_hash() != alternate.get_hash()
    assert AirlineTools(db).get_db_hash() != AirlineTools(alternate).get_db_hash()


@pytest.mark.parametrize(
    "other_db",
    [
        MockDB(tasks={}, users={}),
        RetailDB(products={}, users={}, orders={}),
        TelecomDB(),
    ],
)
def test_other_domain_hashes_are_unchanged(other_db):
    before = other_db.model_dump()
    assert other_db.get_hash() == get_pydantic_hash(other_db)
    assert ToolKitBase(other_db).get_db_hash() == get_pydantic_hash(other_db)
    assert other_db.model_dump() == before


def test_uninitialized_toolkit_hash_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        ToolKitBase().get_db_hash()


@pytest.mark.parametrize("full_duplex", [False, True])
@pytest.mark.parametrize("changed_field", [None, "dob"])
def test_environment_evaluators_use_passenger_equivalence(
    db, full_duplex, changed_field
):
    def constructor(**kwargs):
        return Environment(
            domain_name="airline",
            policy="",
            tools=AirlineTools(db.model_copy(deep=True)),
        )

    passengers = [p.model_dump() for p in db.reservations["BOOK01"].passengers]
    task = Task(
        id="passenger-equivalence",
        user_scenario={"instructions": "Update passengers."},
        evaluation_criteria={
            "reward_basis": ["DB"],
            "actions": [
                {
                    "action_id": "gold",
                    "name": "update_reservation_passengers",
                    "arguments": {"reservation_id": "BOOK01", "passengers": passengers},
                }
            ],
        },
    )
    alternate = [passengers[i].copy() for i in [0, 2, 1]]
    if changed_field:
        alternate[1][changed_field] = "2000-01-01"
    call = ToolCall(
        id="predicted",
        name="update_reservation_passengers",
        arguments={"reservation_id": "BOOK01", "passengers": alternate},
    )
    env = constructor()
    reply = env.get_response(call)
    assert not reply.error
    before = env.tools.db.model_dump()
    reply_before = reply.model_dump()
    env.get_db_hash()
    assert env.tools.db.model_dump() == before
    assert reply.model_dump() == reply_before
    if full_duplex:
        evaluator = FullDuplexEnvironmentEvaluator
        trajectory = [
            Tick(
                tick_id=0,
                timestamp="0",
                agent_tool_calls=[call],
                agent_tool_results=[reply],
            )
        ]
    else:
        evaluator = EnvironmentEvaluator
        trajectory = [AssistantMessage(role="assistant", tool_calls=[call]), reply]
    result = evaluator.calculate_reward(constructor, task, trajectory)
    assert result.reward == (0.0 if changed_field else 1.0)
    assert result.db_check.db_match == (changed_field is None)
