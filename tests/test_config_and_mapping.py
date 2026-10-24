import json

import pytest

from vlm_street_audit.config import PROJECT_ROOT, load_config, load_secret
from vlm_street_audit.mapping import read_segments


def test_local_config_overrides_and_rejects_unknown_settings(tmp_path):
    (tmp_path / "default.config").write_text(json.dumps({"a": 1, "b": 2}))
    (tmp_path / "config").write_text(json.dumps({"b": 3}))
    assert load_config(tmp_path) == {"a": 1, "b": 3}
    (tmp_path / "config").write_text(json.dumps({"typo": 3}))
    with pytest.raises(KeyError, match="typo"):
        load_config(tmp_path)


def test_missing_secret_says_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="default.secret"):
        load_secret(tmp_path)


def test_sample_mapping_reads_into_segments():
    segments = list(read_segments(PROJECT_ROOT / "data" / "city_sample" / "mapping.csv"))
    assert len({(s.locality, s.country) for s in segments}) == 75
    assert all(s.end_second >= s.start_second for s in segments)
    assert {s.time_of_day for s in segments} == {0} and {s.vehicle_type for s in segments} == {0}
