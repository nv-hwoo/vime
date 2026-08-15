import sys
import threading
import types
from argparse import Namespace

import pytest

ray_module = types.ModuleType("ray")
ray_module.get = lambda refs: refs
ray_actor_module = types.ModuleType("ray.actor")
ray_actor_module.ActorHandle = object
sys.modules.setdefault("ray", ray_module)
sys.modules.setdefault("ray.actor", ray_actor_module)

megatron_module = types.ModuleType("megatron")
megatron_core_module = types.ModuleType("megatron.core")
megatron_core_module.mpu = types.SimpleNamespace()
megatron_module.core = megatron_core_module
sys.modules.setdefault("megatron", megatron_module)
sys.modules.setdefault("megatron.core", megatron_core_module)

disk_delta_module = types.ModuleType("vime.utils.disk_delta")
disk_delta_module.make_tensor_reader = lambda path: object()
sys.modules.setdefault("vime.utils.disk_delta", disk_delta_module)

distributed_module = types.ModuleType("vime.utils.distributed_utils")
distributed_module.get_gloo_group = lambda: object()
sys.modules.setdefault("vime.utils.distributed_utils", distributed_module)

base_module = types.ModuleType("vime.backends.megatron_utils.update_weight.update_weight_from_disk_delta")


class UpdateWeightFromDiskDelta:
    def _iter_non_expert_chunks(self):
        return iter(())

    def _iter_expert_chunks(self):
        return iter(())


base_module.UpdateWeightFromDiskDelta = UpdateWeightFromDiskDelta
sys.modules.setdefault(base_module.__name__, base_module)

from vime.backends.megatron_utils.update_weight import update_weight_from_modelexpress as mx_module
from vime.backends.megatron_utils.update_weight.update_weight_from_modelexpress import (
    ModelExpressUpdateError,
    UpdateWeightFromModelExpress,
)

pytestmark = pytest.mark.unit


class RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeCatalog:
    def __init__(self):
        self.commits = []

    def commit_revision(self, model_id, version):
        self.commits.append((model_id, version))


class FakePublisher:
    def __init__(self):
        self.catalog = FakeCatalog()
        self.target_digest = "sha256:launch"
        self.pending_version = None
        self.pending_digest = None
        self.publishes = []
        self.baselines = []
        self.waits = []
        self.metrics = {"perf/mx_publish_time": 5.0}

    def publish_version(self, version, **kwargs):
        self.publishes.append((version, kwargs, threading.current_thread().name))
        self.pending_version = version
        if version != "0":
            self.pending_digest = f"sha256:{version}"

    def capture_baseline(self, gather, read):
        self.baselines.append((gather, read, threading.current_thread().name))

    def wait_for_commit(self, version, completion=None):
        if completion is not None:
            completion.result()
        self.waits.append(version)
        self.pending_version = None
        if version != "0":
            self.target_digest = self.pending_digest
            self.pending_digest = None

    def pop_metrics(self):
        metrics, self.metrics = self.metrics, {}
        return metrics


class FakeEngine:
    def __init__(self, events, update_error=None):
        self.events = events
        self.update_error = update_error
        self.init_weight_transfer_engine = RemoteMethod(self._init)
        self.pause_generation = RemoteMethod(lambda: self._event("pause"))
        self.flush_cache = RemoteMethod(lambda: self._event("flush"))
        self.start_weight_update = RemoteMethod(lambda **kwargs: self._event("start"))
        self.update_weights_from_modelexpress = RemoteMethod(self._update)
        self.finish_weight_update = RemoteMethod(lambda: self._event("finish"))
        self.continue_generation = RemoteMethod(lambda: self._event("continue"))

    def _event(self, name):
        self.events.append((name, threading.current_thread().name))
        return {"ok": True}

    def _init(self, payload):
        self.events.append(("init", payload))
        return {"ok": True}

    def _update(self, target):
        self._event(f"update:{target}")
        if self.update_error:
            raise RuntimeError(self.update_error)
        return {"ok": True}


def args():
    values = dict(
        modelexpress_config={
            "catalog_endpoint": "dns:///catalog:50051",
            "initial_version": "0",
            "model_id": "policy",
            "preparation_cache_dir": "/mxdelta/mxprep",
            "ready_timeout_seconds": 321.0,
            "s3_bucket": "weights",
            "s3_endpoint": "http://minio:9000",
            "future_option": {"enabled": True},
        }
    )
    out = Namespace(**values)
    vars(out)["hf_checkpoint"] = "/models/model"
    return out


@pytest.fixture(autouse=True)
def patch_runtime(monkeypatch):
    monkeypatch.setattr(mx_module.ray, "get", lambda refs: refs)
    monkeypatch.setattr(mx_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(mx_module.dist, "barrier", lambda group=None: None)
    monkeypatch.setattr(mx_module.dist, "broadcast_object_list", lambda values, src, group=None: None)
    monkeypatch.setattr(mx_module.mpu, "get_data_parallel_rank", lambda **kwargs: 0, raising=False)
    monkeypatch.setattr(mx_module.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(mx_module, "get_gloo_group", lambda: object())
    monkeypatch.setattr(mx_module, "make_tensor_reader", lambda _path: object())


def updater(publisher):
    instance = UpdateWeightFromModelExpress(
        args(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3",
        quantization_config=None,
        publisher=publisher,
    )
    instance._for_each_hf_bucket = lambda consume: None
    return instance


def test_vime_initializes_vllm_and_publishes_on_main_thread():
    publisher = FakePublisher()
    instance = updater(publisher)
    events = []
    instance.connect_rollout_engines([FakeEngine(events)], object())

    instance.update_weights()
    instance.update_weights()
    instance._control.shutdown()

    main = threading.current_thread().name
    assert publisher.publishes[0][0] == "0"
    assert publisher.baselines[0][2] == main
    assert publisher.publishes[1] == (
        "1",
        {"base_version": "0", "gather_hf_buckets": instance._for_each_hf_bucket},
        main,
    )
    assert events[0] == (
        "init",
        {
            "init_info": {
                "model_id": "policy",
                "catalog_endpoint": "dns:///catalog:50051",
                "initial_version": "0",
                "preparation_cache_dir": "/mxdelta/mxprep",
                "ready_timeout_seconds": 321.0,
                "s3_endpoint_url": "http://minio:9000",
            }
        },
    )
    control_events = [event for event, _thread in events[1:]]
    assert control_events == ["pause", "flush", "start", "update:1", "finish", "continue"]
    assert all(thread.startswith("modelexpress-control") for _event, thread in events[1:])
    assert publisher.catalog.commits == [("policy", "0"), ("policy", "1")]
    assert publisher.waits == ["0", "1"]
    assert instance.weight_version == 1
    assert instance.pop_metrics() == {"perf/mx_publish_time": 5.0}


def test_reconnecting_the_same_vllm_cohort_is_a_noop():
    publisher = FakePublisher()
    instance = updater(publisher)
    events = []
    engine = FakeEngine(events)

    instance.connect_rollout_engines([engine], object())
    instance.connect_rollout_engines([engine], object())
    instance._control.shutdown()

    assert [event for event, _payload in events if event == "init"] == ["init"]
    assert publisher.catalog.commits == [("policy", "0")]
    assert publisher.waits == ["0"]


def test_failed_vllm_update_is_not_committed_or_resumed():
    publisher = FakePublisher()
    instance = updater(publisher)
    events = []
    instance.connect_rollout_engines([FakeEngine(events, update_error="install failed")], object())
    instance.update_weights()

    with pytest.raises(ModelExpressUpdateError, match="install failed"):
        instance.update_weights()
    instance._control.shutdown()

    assert publisher.catalog.commits == [("policy", "0")]
    assert not any(event == "continue" for event, _thread in events[1:])
    assert instance.weight_version == 0
