from pathlib import Path

import pytest

from kodoom.config import BUILTIN_PROFILES, ProfileError, load_profile


@pytest.mark.parametrize("name", BUILTIN_PROFILES)
def test_builtin_profiles_load(name):
    profile = load_profile(name)
    assert profile.name == name


def test_dev_profile_is_laptop_safe():
    dev = load_profile("dev")
    assert dev.device == "cpu"
    assert dev.precision == "fp32"
    assert dev.threads == 4
    assert dev.max_cases_per_source is not None


def test_colab_profile_uses_full_data_on_drive():
    colab = load_profile("colab")
    assert colab.device == "cuda"
    assert colab.max_cases_per_source is None
    assert colab.runs_dir.as_posix().startswith("/content/drive/")


def test_all_profiles_share_the_seed():
    assert len({load_profile(n).seed for n in BUILTIN_PROFILES}) == 1


def test_colab_writes_datasets_to_drive_and_dev_keeps_them_git_ignored():
    for name in ("colab", "colab-preflight"):
        assert load_profile(name).data_dir.as_posix().startswith("/content/drive/")
    assert load_profile("dev").data_dir == Path("data")  # /data/ is in .gitignore


def test_colab_keeps_models_and_scratch_off_drive():
    for name in ("colab", "colab-preflight"):
        p = load_profile(name)
        assert not p.cache_dir.as_posix().startswith("/content/drive/")
        assert not p.scratch_dir.as_posix().startswith("/content/drive/")
        assert p.runs_dir.as_posix().startswith("/content/drive/")


def test_runs_dir_override():
    assert load_profile("dev", runs_dir="elsewhere").runs_dir == Path("elsewhere")


@pytest.mark.parametrize("name", ["kaggle", "kaggle-preflight"])
def test_kaggle_keeps_caches_out_of_saved_outputs(name):
    profile = load_profile(name)
    assert profile.device == "cuda"
    assert profile.data_dir.as_posix().startswith("/kaggle/working/")
    assert profile.runs_dir.as_posix().startswith("/kaggle/working/")
    assert not profile.cache_dir.as_posix().startswith("/kaggle/working/")
    assert not profile.scratch_dir.as_posix().startswith("/kaggle/working/")


def test_profile_from_file(tmp_path):
    path = tmp_path / "mine.toml"
    path.write_text(
        '[run]\nseed = 7\nruns_dir = "out"\n[compute]\ndevice = "cpu"\nprecision = "fp32"\n'
        '[storage]\nscratch_dir = "s"\ncache_dir = "c"\ndata_dir = "d"\n',
        encoding="utf-8",
    )
    profile = load_profile(path)
    assert (profile.name, profile.seed, profile.threads) == ("mine", 7, None)


def _write(tmp_path, text):
    path = tmp_path / "p.toml"
    path.write_text(text, encoding="utf-8")
    return path


BASE = (
    '[run]\nseed = 1\nruns_dir = "r"\n[compute]\ndevice = "cpu"\nprecision = "fp32"\n'
    '[storage]\nscratch_dir = "s"\ncache_dir = "c"\ndata_dir = "d"\n'
)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (BASE + "[data]\nmax_case_per_source = 3\n", "unknown key"),
        (BASE + "[model]\nname = 'x'\n", "unknown section"),
        (BASE.replace("seed = 1", "seed = true"), "wrong type"),
        (BASE.replace("seed = 1\n", ""), "missing"),
        (BASE.replace('"fp32"', '"fp16"'), "fp16 needs a GPU"),
        (BASE.replace('"cpu"', '"unknown"'), "device must be"),
        (BASE + "[data]\nmax_cases_per_source = 0\n", "at least 1"),
        ("[run\n", "invalid TOML"),
        (BASE.replace('cache_dir = "c"\n', ""), "missing \\[storage\\] cache_dir"),
        (BASE + "reserve_gb = -1\n", "cannot be negative"),
    ],
)
def test_bad_profiles_are_rejected(tmp_path, text, message):
    with pytest.raises(ProfileError, match=message):
        load_profile(_write(tmp_path, text))


def test_unknown_profile_name():
    with pytest.raises(ProfileError, match="unknown profile"):
        load_profile("laptop")
