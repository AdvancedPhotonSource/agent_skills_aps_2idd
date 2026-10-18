"""Synchronous OSA alignment adapter for the APS 2-ID-D QueueServer."""
from __future__ import annotations

import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
from uuid import uuid4


class APS2IDDAdapter:
    """Use absolute OSA coordinates in micrometers and shared TIFF paths."""

    units = "um"

    def __init__(self, config, *, client=None):
        url = config.get("url", "")
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https", "tcp") or not parsed.hostname:
            raise ValueError("Set adapter config 'url' to the QueueServer HTTP URL or tcp://host:port")
        self.request_timeout = float(config.get("request_timeout", 5.))
        self.operation_timeout = float(config.get("operation_timeout", 60.))
        self.motion_timeout = float(config.get("motion_timeout", 30.))
        self.acquisition_timeout = float(config.get("acquisition_timeout", 6.))
        self.settle_time = float(config.get("settle_time", .2))
        self.poll_interval = float(config.get("poll_interval", .2))
        # Optional dwell after every adapter call, for instruments that need recovery time.
        self.post_call_sleep = float(config.get("post_call_sleep", 0.))
        positive = (self.request_timeout, self.operation_timeout, self.motion_timeout,
                    self.acquisition_timeout, self.poll_interval)
        if (not all(math.isfinite(v) and v > 0 for v in positive)
            or not all(math.isfinite(v) and v >= 0 for v in (self.settle_time, self.post_call_sleep))
            or self.operation_timeout <= max(self.motion_timeout + self.settle_time,
                                             self.acquisition_timeout)):
            raise ValueError("Require finite positive timeouts and an operation timeout longer than motion/acquisition")
        self.identity = dict(user=config.get("user", "user2idd"),
                             user_group=config.get("user_group", "root"))
        if client is None:
            if parsed.scheme == "tcp":
                from bluesky_queueserver_api.zmq import REManagerAPI
                # client = REManagerAPI(zmq_control_addr=url, timeout_recv=self.request_timeout,
                #                       timeout_send=self.request_timeout)
                client = REManagerAPI()
            else:
                from bluesky_queueserver_api.http import REManagerAPI
                client = REManagerAPI(http_server_uri=url, timeout=self.request_timeout)
                if config.get("api_key_env"):
                    client.set_authorization_key(api_key=os.environ[config["api_key_env"]])
        self.client = client

    @staticmethod
    def _checked(response):
        if not response.get("success"):
            raise RuntimeError(f"QueueServer request failed: {response.get('msg', response)}")
        return response

    def _pause(self):
        if self.post_call_sleep > 0:
            time.sleep(self.post_call_sleep)

    def _wait(self, deadline, label):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for {label}; check QueueServer before further operations")
        time.sleep(min(self.poll_interval, remaining))

    def _function(self, name, **kwargs):
        deadline = time.monotonic() + self.operation_timeout
                
        response = self._checked(self.client.function_execute(
            dict(item_type="function", name=name, kwargs=kwargs),
            run_in_background=False, **self.identity))
        if not response["success"]:
            raise RuntimeError(response["msg"])
        uid = response["task_uid"]
        while True:
            response = self._checked(self.client.task_result(uid))
            if response["status"] == "completed":
                result = response["result"]
                if not result["success"]:
                    raise RuntimeError(f"{name} failed: {result.get('msg')}")
                return result["return_value"]
            if response["status"] != "running":
                raise RuntimeError(f"{name} task {uid}: {response['status']}")
            self._wait(deadline, f"{name} task {uid}")

    def read_position(self):
        """Read the worker's millimeter coordinates and return [x, y] in µm."""
        position = self._function("read_osa_position")
        if len(position) != 2 or not all(math.isfinite(float(v)) for v in position):
            raise RuntimeError("OSA readback must contain two finite coordinates")
        self._pause()
        return [float(v) * 1000 for v in position]

    def move_absolute(self, position):
        """Execute one move and wait for its successful history entry, including settling.
        Expected inputs are in micron. Values in sent requests are converted to mm.
        """
        if len(position) != 2 or not all(math.isfinite(float(v)) for v in position):
            raise ValueError("OSA target must contain two finite coordinates")
        deadline = time.monotonic() + self.operation_timeout
        response = self._checked(self.client.item_execute(
            dict(item_type="plan", name="move_osa_absolute", kwargs=dict(
                x_mm=float(position[0]) / 1000, y_mm=float(position[1]) / 1000,
            )), **self.identity))
        uid = response["item"]["item_uid"]
        while True:
            history = self._checked(self.client.history_get(reload=True))
            for item in reversed(history["items"]):
                if item["item_uid"] == uid:
                    result = item["result"]
                    if result["exit_status"] != "completed":
                        raise RuntimeError(f"OSA move {uid} {result['exit_status']}: {result.get('msg', '')}")
                    self._pause()
                    return
            status = self.client.status(reload=True)
            if status.get("manager_state") == "paused":
                raise RuntimeError(f"QueueServer paused during OSA move {uid}; operator action required")
            self._wait(deadline, f"OSA move {uid}")

    def acquire_image(self):
        """Acquire a fresh X-eye frame and return its locally accessible saved path."""
        result = self._function("acquire_xeye_image", filename=f"osa_alignment_{uuid4().hex}.tiff",
                                timeout=self.acquisition_timeout)
        if not result.get("success"):
            raise RuntimeError(f"X-eye image acquisition failed: {result.get('error')}")
        path = Path(result["img_path"])
        if not path.is_absolute() or not path.is_file():
            raise FileNotFoundError(f"X-eye image must be accessible through a shared absolute path: {path}")
        self._pause()
        return str(path)

    def close(self):
        self.client.close()


def create_adapter(config):
    return APS2IDDAdapter(config)
