"""Invented small English inputs for the CPU restart smoke run."""

from dataclasses import replace

from kodoom.bulk.state import Unit
from kodoom.schema import Option, Record


def fixture_units() -> list[Unit]:
    units = []
    for index, split in enumerate(("train", "test")):
        first = Record(
            id=f"fixture-{index}:q0",
            source_id=f"fixture-{index}",
            source="LocalLLaMA/typed-decisions",
            source_revision="fixture",
            license="Apache-2.0",
            split=split,
            origin="synthetic",
            task_family="workflow-customer-service",
            state_lang="en",
            question_lang="en",
            state='{"thread":[{"role":"customer","text":"Order A-123 is 3 days late."}]}',
            question_type="noul",
            question_text="Is the order late?",
            options=(Option("false", "No"), Option("true", "Yes")),
            gold={"false": 0.0, "true": 1.0},
            extra={"workflow": "customer_service", "question": "q0"},
        )
        units.append(
            Unit(
                "typed-decisions",
                tuple(
                    replace(
                        first,
                        id=f"fixture-{index}:q{n}",
                        extra={**first.extra, "question": f"q{n}"},
                    )
                    for n in range(5)
                ),
            )
        )
    for index in range(3):
        record = Record(
            id=f"helmo-fixture-{index}",
            source_id=f"helmo-fixture-{index}",
            source="helmo/synthetic-typed-decisions",
            source_revision="fixture",
            license="MIT",
            split="train",
            origin="synthetic",
            task_family="topics",
            state_lang="en",
            question_lang="en",
            state="An order is 3 days late.",
            question_type="noul",
            question_text="Is it late?",
            options=(Option("false", "No"), Option("true", "Yes")),
            gold={"false": 0.0, "true": 1.0},
            extra={"topic": f"topic-{index}"},
        )
        units.append(Unit("helmo", (record,)))
    return units
