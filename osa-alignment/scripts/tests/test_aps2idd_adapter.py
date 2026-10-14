"""QueueServer contract and simulated OSA tests; no beamline connections."""
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aps2idd_adapter import APS2IDDAdapter, create_adapter
from aps2idd_qserver import move_osa_absolute, read_osa_position


@pytest.fixture
def client():
    api = pytest.importorskip("bluesky_queueserver_api.http")
    client = create_autospec(api.REManagerAPI, instance=True)
    client.function_execute.return_value = dict(success=True, task_uid="task-1")
    client.item_execute.return_value = dict(success=True, item=dict(item_uid="move-1"))
    client.status.return_value = dict(manager_state="executing_queue")
    return client


@pytest.fixture
def adapter(client):
    return APS2IDDAdapter(dict(url="http://offline.invalid:60610", poll_interval=.001), client=client)


def task(value=None, **result):
    return dict(success=True, status="completed",
                result=dict(dict(success=True, return_value=value), **result))


def history(uid="move-1", status="completed"):
    return dict(success=True, items=[dict(item_uid=uid, result=dict(exit_status=status, msg="detail"))])


def test_placeholder_rejected_before_client_creation():
    config = json.loads((Path(__file__).resolve().parents[1] / "aps2idd.adapter.json").read_text())
    with pytest.raises(ValueError, match="url"):
        create_adapter(config)


@pytest.mark.parametrize("option,value", [("motion_timeout", 0), ("settle_time", -1),
                                         ("request_timeout", float("inf")), ("operation_timeout", 20)])
def test_invalid_timeouts(option, value, client):
    with pytest.raises(ValueError, match="timeout"):
        APS2IDDAdapter(dict(url="tcp://offline.invalid:60615", **{option: value}), client=client)


@pytest.mark.parametrize("url,module,expected", [
    ("http://offline.invalid:60610", "http", dict(http_server_uri="http://offline.invalid:60610", timeout=5.)),
    ("tcp://offline.invalid:60615", "zmq", dict(zmq_control_addr="tcp://offline.invalid:60615",
                                             timeout_recv=5., timeout_send=5.)),
])
def test_client_transport(monkeypatch, url, module, expected):
    api = pytest.importorskip(f"bluesky_queueserver_api.{module}")
    constructor = Mock()
    monkeypatch.setattr(api, "REManagerAPI", constructor)
    create_adapter(dict(url=url))
    constructor.assert_called_once_with(**expected)


def test_read_position_waits_and_converts_mm_to_um(adapter, client):
    client.task_result.side_effect = [dict(success=True, status="running"), task([1.25, -.003])]
    assert adapter.read_position() == [1250., -3.]
    client.function_execute.assert_called_once_with(
        dict(item_type="function", name="read_osa_position", kwargs={}),
        run_in_background=False, user="osa-alignment", user_group="primary")
    assert client.task_result.call_count == 2


@pytest.mark.parametrize("response,match", [
    (task(success=False, msg="disconnected"), "disconnected"),
    (dict(success=True, status="not_found"), "not_found"),
    (dict(success=False, msg="request denied"), "request denied"),
    (task([float("nan"), 0]), "finite"),
])
def test_bad_readbacks_or_tasks_raise(adapter, client, response, match):
    client.task_result.return_value = response
    with pytest.raises(RuntimeError, match=match):
        adapter.read_position()


def test_move_matches_own_uid_and_converts_um_to_mm(adapter, client):
    client.history_get.side_effect = [history("another-move"), history()]
    adapter.move_absolute([1250., -3.])
    client.item_execute.assert_called_once_with(
        dict(item_type="plan", name="move_osa_absolute", kwargs=dict(
            x_mm=1.25, y_mm=-.003, timeout=30., settle_time=.2)),
        user="osa-alignment", user_group="primary")
    assert client.history_get.call_count == 2
    client.item_add.assert_not_called()
    client.queue_start.assert_not_called()


@pytest.mark.parametrize("status", ["failed", "stopped", "aborted", "halted"])
def test_unsuccessful_move_raises(adapter, client, status):
    client.history_get.return_value = history(status=status)
    with pytest.raises(RuntimeError, match=status):
        adapter.move_absolute([0, 0])


def test_paused_move_raises(adapter, client):
    client.history_get.return_value = history("another-move")
    client.status.return_value = dict(manager_state="paused")
    with pytest.raises(RuntimeError, match="paused"):
        adapter.move_absolute([0, 0])


def test_rejected_move_not_retried(adapter, client):
    client.item_execute.return_value = dict(success=False, msg="worker busy")
    with pytest.raises(RuntimeError, match="worker busy"):
        adapter.move_absolute([0, 0])
    assert client.item_execute.call_count == 1
    client.history_get.assert_not_called()


@pytest.mark.parametrize("method", ["read_position", "move_absolute"])
def test_polling_timeout(adapter, client, monkeypatch, method):
    clock = Mock(side_effect=[0., 61.])
    monkeypatch.setattr("aps2idd_adapter.time.monotonic", clock)
    client.task_result.return_value = dict(success=True, status="running")
    client.history_get.return_value = history("another-move")
    with pytest.raises(TimeoutError, match="check QueueServer"):
        getattr(adapter, method)(*([[0, 0]] if method == "move_absolute" else []))


def test_images_have_unique_names_and_accessible_paths(adapter, client, tmp_path):
    path = tmp_path / "snapshot.tiff"
    path.touch()
    client.task_result.return_value = task(dict(success=True, img_path=str(path)))
    assert adapter.acquire_image() == str(path)
    assert adapter.acquire_image() == str(path)
    calls = client.function_execute.call_args_list
    assert calls[0].args[0]["kwargs"]["filename"] != calls[1].args[0]["kwargs"]["filename"]
    assert calls[0].args[0]["kwargs"]["timeout"] == 6.


@pytest.mark.parametrize("value,error", [
    (dict(success=False, error="camera offline"), RuntimeError),
    (dict(success=True, img_path="/unmounted/osa/frame.tiff"), FileNotFoundError),
])
def test_acquisition_failure(adapter, client, value, error):
    client.task_result.return_value = task(value)
    with pytest.raises(error):
        adapter.acquire_image()


@pytest.fixture
def motors(monkeypatch):
    sim = pytest.importorskip("ophyd.sim")
    osa = SimpleNamespace(x=sim.SynAxis(name="osa_x"), y=sim.SynAxis(name="osa_y"))
    # Supply explicit EPICS-style limit checks for the simulated axes.
    osa.x.check_value = Mock()
    osa.y.check_value = Mock()
    module = ModuleType("apsbits.core.instrument_init")
    module.oregistry = dict(osa=osa)
    monkeypatch.setitem(sys.modules, "apsbits.core.instrument_init", module)
    return osa


def test_worker_move_and_readback_with_runengine(motors):
    bluesky = pytest.importorskip("bluesky")
    engine = bluesky.RunEngine({})
    messages = []
    engine.msg_hook = messages.append
    engine(move_osa_absolute(1.25, -.003, timeout=1., settle_time=.001))
    assert read_osa_position() == [1.25, -.003]
    assert [msg.command for msg in messages] == ["set", "set", "wait", "sleep"]
    assert messages[2].kwargs["timeout"] == 1.


def test_worker_checks_both_limits_before_moving(motors):
    motors.y.check_value.side_effect = ValueError("outside y limits")
    with pytest.raises(ValueError, match="outside y limits"):
        next(move_osa_absolute(1., 2.))
    assert read_osa_position() == [0., 0.]


def test_worker_stops_axes_on_motion_timeout(motors):
    bluesky = pytest.importorskip("bluesky")
    from ophyd.status import Status

    motors.x.set = Mock(return_value=Status())  # Never completes.
    motors.x.stop = Mock()
    motors.y.stop = Mock()
    with pytest.raises(TimeoutError):
        bluesky.RunEngine({})(move_osa_absolute(1., 2., timeout=.01, settle_time=0.))
    motors.x.stop.assert_called()
    motors.y.stop.assert_called()
