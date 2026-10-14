"""Dashboard integration checks; all acquisition uses a synthetic backend."""
import json
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import align_osa
from alignment_webui import Dashboard
from test_analytical_alignment import SyntheticBackend, calibration, observation, settings


def request(dashboard, path="api/status", method="GET"):
    return urlopen(Request(dashboard.url + path, method=method), timeout=5)


def test_live_updates_partial_records_and_read_only_routes(tmp_path):
    backend = SyntheticBackend([0, 0])
    image = tmp_path / "detector.tif"
    Image.fromarray(np.arange(400, dtype=np.uint16).reshape(20, 20)).save(image)

    def measurement(p, c):
        obs = observation(p, invalid=backend.acquisitions == 1)
        obs["image"] = str(image)
        return obs

    controller = align_osa.Controller(calibration(), settings(), backend, tmp_path / "run.jsonl", measurement)
    dashboard = Dashboard(controller.log_path, port=0)
    dashboard.start()
    try:
        initial = json.load(request(dashboard))
        assert initial["observations"] == []
        assert initial["settings"]["max_acquisitions"] == 40
        controller.acquire([0, 0], "Baseline", "coarse")
        first = json.load(request(dashboard))
        assert len(first["observations"]) == 1
        assert first["observations"][0]["metrics"]["primary_metric"] == "inf"
        assert first["image"].startswith("data:image/png;base64,")
        controller.acquire([1, 0], "Next trial", "bo")
        second = json.load(request(dashboard))
        assert len(second["observations"]) == 2
        assert second["observations"][-1]["position"] == [1, 0]
        with controller.log_path.open("a") as stream:
            stream.write('{"event":')
        assert json.load(request(dashboard))["observations"] == second["observations"]
        counts = (len(backend.moves), backend.acquisitions)
        for path, method, status in [("api/status", "POST", 501), ("api/status", "PUT", 501),
                                     ("api/status", "DELETE", 501), ("../calibration.json", "GET", 404),
                                     ("api/status?image=/etc/passwd", "GET", 404)]:
            with pytest.raises(HTTPError) as error:
                request(dashboard, path, method)
            assert error.value.code == status
        assert counts == (len(backend.moves), backend.acquisitions)
        assert b"Search trajectory" in request(dashboard, "").read()
    finally:
        dashboard.close()
    assert not dashboard.thread.is_alive()
    assert dashboard.snapshot_path.exists()


@pytest.mark.parametrize("blocked", [False, True])
def test_cli_flag_preserves_results_and_saves_final_snapshot(tmp_path, monkeypatch, capsys, blocked):
    outputs = []
    for enabled in (False, True):
        folder = tmp_path / str(enabled)
        folder.mkdir()
        (folder / "cal.json").write_text(json.dumps(calibration()))
        (folder / "config.json").write_text(json.dumps({"units": "um", "bounds": [[-15, 15], [-15, 15]],
            "max_step": 5, "target_m2": 10, "max_acquisitions": 8}))
        backend = SyntheticBackend([0, 0])
        if blocked:
            def fail_acquisition():
                raise RuntimeError("Synthetic acquisition failure")
            backend.acquire_image = fail_acquisition
        monkeypatch.setattr(align_osa.importlib, "import_module", lambda name: type("Adapter", (), {
            "create": staticmethod(lambda options: backend)}))
        monkeypatch.setattr(align_osa, "measure", lambda p, c, **kwargs: observation(p))
        argv = ["align_osa.py", "--calibration", str(folder / "cal.json"), "--config", str(folder / "config.json"),
                "--adapter", "synthetic:create", "--log", str(folder / "run.jsonl"), "--output", str(folder / "result.json")]
        if enabled:
            argv += ["--webui", "--webui-port", "0"]
        monkeypatch.setattr(sys, "argv", argv)
        if blocked:
            with pytest.raises(SystemExit) as exit_info:
                align_osa.main()
            assert exit_info.value.code == 2
        else:
            align_osa.main()
        result = json.loads((folder / "result.json").read_text())
        result.pop("log")
        outputs.append(result)
        snapshot = folder / "run.html"
        assert snapshot.exists() == enabled
        if enabled:
            html = snapshot.read_text()
            expected_status = "blocked" if blocked else "metric_target_reached"
            assert f'"status": "{expected_status}"' in html
            assert '"final_readback": [0.0, 0.0]' in html
            assert "OSA web UI: http://127.0.0.1:" in capsys.readouterr().out
    assert outputs[0] == outputs[1]


def test_snapshot_escapes_log_content_and_preserves_existing_file(tmp_path):
    controller = align_osa.Controller(calibration(), settings(), SyntheticBackend([0, 0]), tmp_path / "run.jsonl")
    controller.record("fault", text="</script><script>alert('test')</script>")
    dashboard = Dashboard(controller.log_path, port=0)
    assert "</script><script>" not in dashboard.html(dashboard.state())
    dashboard.snapshot_path.write_text("existing report")
    dashboard.start()
    dashboard.close()
    assert dashboard.snapshot_path.read_text() == "existing report"
