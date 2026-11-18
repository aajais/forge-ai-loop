import json

from bufogen.evaluation import calibration
from bufogen.evaluation.scorers.astra_judge import AstraReferenceJudge, config_key


def _cfg(**kw):
    return AstraReferenceJudge(rubric_version="bufo-astra-v5", samples=3, **kw).configuration()


def test_v5_k3_is_calibrated_and_v1_is_not():
    assert calibration.status(_cfg()) == "calibrated"
    assert calibration.status(AstraReferenceJudge().configuration()) == "uncalibrated"
    k1 = AstraReferenceJudge(rubric_version="bufo-astra-v5", samples=1).configuration()
    assert calibration.status(k1) == "uncalibrated"
    assert calibration.record(AstraReferenceJudge().configuration()) is None


def test_any_configuration_difference_is_uncalibrated():
    assert calibration.status(_cfg(threshold=0.99)) == "uncalibrated"
    assert calibration.status(_cfg(confidence_floor=0.3)) == "uncalibrated"
    assert calibration.status(_cfg(judge_model="gpt-6-other")) == "uncalibrated"
    for field, value in (
        ("max_completion_tokens", 1),
        ("image_detail", "low"),
        ("rubric_hash", "x"),
    ):
        assert calibration.status(dict(_cfg(), **{field: value})) == "uncalibrated"


def test_registry_config_matches_the_calibration_record():
    record = calibration.record(_cfg())
    judge = record["judge"]
    stored = calibration.CALIBRATED_CONFIGS[("bufo-astra-v5", 3)]
    for key, value in judge.items():
        assert stored[key] == value


def test_record_requires_gate_passed(tmp_path, monkeypatch):
    path = tmp_path / "record.json"
    path.write_text(json.dumps({"gate": {"passed": False}}))
    monkeypatch.setitem(calibration.CALIBRATED, ("bufo-astra-v5", 3), str(path))
    assert calibration.status(_cfg()) == "uncalibrated"
    assert calibration.record(_cfg()) is None
    path.write_text(json.dumps({"gate": {"passed": True}}))
    assert calibration.status(_cfg()) == "calibrated"


def test_v1_rows_still_replay_against_derived_status():
    live = AstraReferenceJudge().configuration()
    assert live["calibration_status"] == "uncalibrated"
    saved = dict(live, calibration_status="uncalibrated")
    assert config_key(saved) == config_key(live)


def test_v5_k3_rows_replay_after_status_change():
    live = AstraReferenceJudge(rubric_version="bufo-astra-v5", samples=3).configuration()
    assert live["calibration_status"] == "calibrated"
    saved = dict(live, calibration_status="uncalibrated")
    assert saved != live
    assert config_key(saved) == config_key(live)
    assert "calibration_status" not in config_key(live)
