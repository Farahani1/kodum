import pytest

from kodoom.schema import Option, Record
from kodoom.sources import (
    EXCLUDED,
    SOURCES,
    TEST_ONLY,
    SourceError,
    check_record,
    get_source,
)


def record(source, split="train", license=None):
    return Record(
        id="r1",
        source_id="c1",
        source=source,
        source_revision=None,
        license=license or SOURCES[source].license,
        split=split,
        origin="native",
        task_family="reading",
        state_lang="fa",
        question_lang="fa",
        state="متن",
        question_type="noul",
        question_text="درست است؟",
        options=(Option("yes", "بله"), Option("no", "خیر")),
        gold={"yes": 1.0},
    )


def test_plan_roles():
    assert {n for n, s in SOURCES.items() if s.role == TEST_ONLY} == {
        "persiannlp/parsinlu",
        "sajjjadayobi/PersianQA",
        "facebook/belebele",
        "own/stt-intent",
    }
    assert {n for n, s in SOURCES.items() if s.role == EXCLUDED} == {
        "raia-center/khayyam-challenge"
    }


@pytest.mark.parametrize("split", ["train", "validation", "calibration"])
@pytest.mark.parametrize(
    "source",
    ["persiannlp/parsinlu", "sajjjadayobi/PersianQA", "facebook/belebele", "own/stt-intent"],
)
def test_test_only_sources_never_enter_training(source, split):
    with pytest.raises(SourceError, match="test-only"):
        check_record(record(source, split))
    check_record(record(source, "test"))


def test_excluded_source_is_rejected_in_any_split():
    with pytest.raises(SourceError, match="excluded"):
        check_record(record("raia-center/khayyam-challenge", "test"))


def test_license_must_match_the_source():
    with pytest.raises(SourceError, match="does not match"):
        check_record(record("sajjjadayobi/PersianQA", split="test", license="Apache-2.0"))


def test_unknown_source():
    with pytest.raises(SourceError, match="unknown source"):
        get_source("someone/else")


def test_gpl_and_nc_data_is_not_published():
    assert not SOURCES["sajjjadayobi/PersianQA"].publish_derived_data
    assert not SOURCES["persiannlp/parsinlu"].publish_derived_data


def test_every_training_source_has_a_permissive_license():
    permissive = {"Apache-2.0", "MIT", "CC-BY-4.0"}
    trainable = [s for s in SOURCES.values() if s.role == "train_and_test"]
    assert {s.license for s in trainable} <= permissive


def test_licenses_confirmed_against_their_repositories():
    # MASSIVE: NOTICE.md; Belebele: README; both checked in September 2026.
    assert SOURCES["alexa/massive"].license_confirmed
    assert SOURCES["facebook/belebele"].license_confirmed
