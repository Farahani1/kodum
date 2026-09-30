"""Every data source the plan names, with its license and allowed role.

The plan's license and leakage rules live here as data, and ``check_record``
enforces them, so a converter cannot put a test-only source into training or
publish a record under the wrong license by mistake.
"""

from __future__ import annotations

from dataclasses import dataclass

from kodoom.schema import Record

TRAIN_AND_TEST = "train_and_test"
TEST_ONLY = "test_only"
EXCLUDED = "excluded"


class SourceError(ValueError):
    """A record breaks a source's license or role."""


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    license: str
    role: str
    # False where the plan marks the license "from memory; confirm it".
    license_confirmed: bool = True
    # False where only the converter may be published, not converted data.
    publish_derived_data: bool = True


SOURCES: dict[str, Source] = {
    s.name: s
    for s in (
        Source(
            "LocalLLaMA/typed-decisions",
            "https://huggingface.co/datasets/LocalLLaMA/typed-decisions",
            "Apache-2.0",
            TRAIN_AND_TEST,
        ),
        Source(
            "helmo/synthetic-typed-decisions",
            "https://huggingface.co/datasets/helmo/synthetic-typed-decisions",
            "MIT",
            TRAIN_AND_TEST,
        ),
        Source(
            "kodoom/code-labeled",
            "https://github.com/Farahani1/kodum",
            "Apache-2.0",
            TRAIN_AND_TEST,
        ),
        Source(
            "alexa/massive",
            "https://github.com/alexa/massive",
            "CC-BY-4.0",
            TRAIN_AND_TEST,
            license_confirmed=False,
        ),
        Source(
            "dml-qom/FarsTail",
            "https://github.com/dml-qom/FarsTail",
            "Apache-2.0",
            TRAIN_AND_TEST,
        ),
        Source(
            "sajjjadayobi/PersianQA",
            "https://github.com/sajjjadayobi/PersianQA",
            "GPL-3.0",
            TRAIN_AND_TEST,
            publish_derived_data=False,
        ),
        Source(
            "persiannlp/parsinlu",
            "https://huggingface.co/persiannlp",
            "CC-BY-NC-SA-4.0",
            TEST_ONLY,
            publish_derived_data=False,
        ),
        Source(
            "facebook/belebele",
            "https://huggingface.co/datasets/facebook/belebele",
            "CC-BY-SA-4.0",
            TEST_ONLY,
            license_confirmed=False,
        ),
        Source("own/stt-intent", "", "private", TEST_ONLY, publish_derived_data=False),
        Source(
            "raia-center/khayyam-challenge",
            "https://github.com/raia-center/khayyam-challenge",
            "CC-BY-ND (academic only)",
            EXCLUDED,
            publish_derived_data=False,
        ),
    )
}


def get_source(name: str) -> Source:
    try:
        return SOURCES[name]
    except KeyError:
        raise SourceError(f"unknown source {name!r}; register it in kodoom.sources") from None


def check_record(record: Record) -> None:
    """Raise if the record's source, license or split breaks the plan's rules."""
    source = get_source(record.source)
    if source.role == EXCLUDED:
        raise SourceError(f"record {record.id!r}: {source.name} is excluded from the project")
    if source.role == TEST_ONLY and record.split != "test":
        raise SourceError(
            f"record {record.id!r}: {source.name} is test-only but the record is in "
            f"split {record.split!r}"
        )
    if record.license != source.license:
        raise SourceError(
            f"record {record.id!r}: license {record.license!r} does not match "
            f"{source.name}'s {source.license!r}"
        )
