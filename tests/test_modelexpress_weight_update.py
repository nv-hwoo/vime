import sys
import types
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

import pytest
import torch

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


class WeightPayloadFormat(Enum):
    XOR_DELTA = "XOR_DELTA"


class WeightVersionState(Enum):
    STAGING = "STAGING"
    READY = "READY"


class ObjectStorageType(Enum):
    S3 = "S3"


class ObjectStorageConfig(SimpleNamespace):
    def root_uri(self, version_number):
        return f"{self.uri_prefix.rstrip('/')}/v{version_number}/model.safetensors.index.json"


@dataclass(frozen=True)
class ObjectStorageSource:
    storage_type: ObjectStorageType
    uri: str


@dataclass(frozen=True)
class WeightVersionRef:
    version_id: str


modelexpress_rl = types.ModuleType("modelexpress_rl")
modelexpress_rl.ModelExpressControlClient = object
modelexpress_rl.ModelExpressTrainerClient = object
modelexpress_rl.ModelExpressTrainerConfig = object
modelexpress_rl.ObjectStorageConfig = ObjectStorageConfig
modelexpress_rl.ObjectStorageSource = ObjectStorageSource
modelexpress_rl.ObjectStorageType = ObjectStorageType
modelexpress_rl.TrainerStagingMode = SimpleNamespace(WRITE_TO_STORAGE="WRITE_TO_STORAGE")
modelexpress_rl.WeightPayloadFormat = WeightPayloadFormat
modelexpress_rl.WeightVersionRef = WeightVersionRef
modelexpress_rl.WeightVersionState = WeightVersionState
sys.modules.setdefault("modelexpress_rl", modelexpress_rl)

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


class FakeControl:
    def __init__(self):
        self.created = []
        self.state_updates = []
        self.state_error = None

    def create_weight_version(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(version_id=f"uid-{kwargs['version_number']}")

    def update_weight_version_state(self, version_id, state):
        self.state_updates.append((version_id, state))
        if self.state_error is not None:
            error, self.state_error = self.state_error, None
            raise RuntimeError(error)
        return SimpleNamespace(version_id=version_id, state=state)


class FakeStaged:
    def __init__(self, trainer, version_id, buckets):
        self._trainer = trainer
        self._version_id = version_id
        self._buckets = buckets

    def publish(self):
        self._trainer.publishes.append((self._version_id, self._buckets))


class FakeTrainer:
    def __init__(self):
        self.baselines = []
        self.stages = []
        self.publishes = []
        self.metrics = {
            "changed_bytes": 25,
            "total_bytes": 100,
            "wire_bytes": 123,
            "stage_delta_time": 7.0,
            "publish_object_storage_time": 8.0,
        }

    def prepare_delta_base(self, *, hf_tensor_iter):
        self.baselines.append(list(hf_tensor_iter))

    def stage_shard(self, *, version, hf_tensor_iter):
        buckets = list(hf_tensor_iter)
        self.stages.append((version.version_id, buckets))
        return FakeStaged(self, version.version_id, buckets)

    def pop_metrics(self):
        metrics, self.metrics = self.metrics, {}
        return metrics


class FakeEngine:
    def __init__(self, events, update_error=None, init_error=None):
        self.events = events
        self.update_error = update_error
        self.init_error = init_error
        self.active = False
        self.init_weight_transfer_engine = RemoteMethod(self._init)
        self.pause_generation = RemoteMethod(lambda: self._event("pause"))
        self.flush_cache = RemoteMethod(lambda: self._event("flush"))
        self.start_weight_update = RemoteMethod(self._start)
        self.update_weights_from_modelexpress = RemoteMethod(self._update)
        self.finish_weight_update = RemoteMethod(self._finish)
        self.continue_generation = RemoteMethod(lambda: self._event("continue"))

    def _event(self, name):
        self.events.append((name, None))
        return {"ok": True}

    def _init(self, payload):
        self.events.append(("init", payload))
        if self.init_error:
            raise RuntimeError(self.init_error)
        return {"ok": True}

    def _start(self):
        if self.active:
            raise RuntimeError("already active")
        self.active = True
        return self._event("start")

    def _update(self, version_id):
        self._event(f"update:{version_id}")
        if self.update_error:
            self.active = False
            raise RuntimeError(self.update_error)
        return {"ok": True}

    def _finish(self, version=None):
        if not self.active:
            raise RuntimeError("not active")
        self.active = False
        return self._event(f"finish:{version}")


def args():
    values = dict(
        modelexpress_config={
            "model_name": "policy",
            "server_url": "dns:///mx:50051",
            "initial_base_version_id": "base-uid",
            "initial_version": 0,
            "preparation_cache_dir": "/mxdelta/mxprep",
            "s3_uri_prefix": "s3://weights/run/policy",
            "s3_endpoint_url": "http://minio:9000",
            "s3_region_name": "us-west-2",
            "rpc_timeout_seconds": 321.0,
            "max_transfer_attempts": 4,
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
    monkeypatch.setattr(
        mx_module.dist,
        "all_reduce",
        lambda value, op=None, group=None: None,
    )
    monkeypatch.setattr(
        mx_module.dist,
        "broadcast_object_list",
        lambda values, src, group=None: None,
    )
    monkeypatch.setattr(
        mx_module.mpu,
        "get_data_parallel_rank",
        lambda **_kwargs: 0,
        raising=False,
    )
    monkeypatch.setattr(
        mx_module.mpu,
        "get_tensor_model_parallel_rank",
        lambda: 0,
        raising=False,
    )
    monkeypatch.setattr(mx_module, "get_gloo_group", lambda: object())


def updater(control, trainer):
    instance = UpdateWeightFromModelExpress(
        args(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3",
        quantization_config=None,
        control_client=control,
        trainer_client=trainer,
    )
    instance._iter_hf_buckets = lambda: iter([[("weight", object())]])
    return instance


def test_vime_builds_generic_trainer_object_storage_config(monkeypatch):
    captured = {}

    class Config(SimpleNamespace):
        pass

    class TrainerClient:
        @staticmethod
        def initialize(config):
            captured["config"] = config
            return FakeTrainer()

    monkeypatch.setattr(modelexpress_rl, "ModelExpressTrainerClient", TrainerClient)
    monkeypatch.setattr(modelexpress_rl, "ModelExpressTrainerConfig", Config)

    UpdateWeightFromModelExpress(
        args(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3",
        quantization_config=None,
        control_client=FakeControl(),
    )

    config = captured["config"]
    assert config.object_storage.storage_type is ObjectStorageType.S3
    assert config.object_storage.uri_prefix == "s3://weights/run/policy"
    assert not hasattr(config.object_storage, "process_group")
    assert config.process_group is not None


def test_vime_initializes_vllm_and_publishes_version_owned_s3_delta(monkeypatch):
    control = FakeControl()
    trainer = FakeTrainer()
    instance = updater(control, trainer)
    events = []
    instance.connect_rollout_engines([FakeEngine(events)], object())

    instance.update_weights()
    clock = iter([0.0, 1.0, 10.0, 12.0, 20.0, 23.0, 30.0, 34.0, 40.0, 45.0])
    reductions = []
    monkeypatch.setattr(mx_module, "perf_counter", lambda: next(clock))

    def reduce_metrics(value, op=None, group=None):
        reductions.append((value.tolist(), op))
        if value.dtype == torch.int64:
            value.copy_(torch.tensor([50, 200, 246], dtype=value.dtype))
        else:
            value.copy_(
                torch.tensor(
                    [17.0, 18.0, 11.0, 12.0, 13.0, 14.0, 15.0],
                    dtype=value.dtype,
                )
            )

    monkeypatch.setattr(mx_module.dist, "all_reduce", reduce_metrics)
    instance.update_weights()

    assert trainer.baselines
    assert trainer.stages[0][0] == "uid-1"
    assert trainer.publishes[0][0] == "uid-1"
    assert control.created == [
        {
            "model_name": "policy",
            "version_number": 1,
            "idempotency_key": "vime:base-uid:v1",
            "payload_format": WeightPayloadFormat.XOR_DELTA,
            "base_version_id": "base-uid",
            "object_storage": ObjectStorageSource(
                storage_type=ObjectStorageType.S3,
                uri="s3://weights/run/policy/v1/model.safetensors.index.json",
            ),
            "state": WeightVersionState.STAGING,
        }
    ]
    assert control.state_updates == [("uid-1", WeightVersionState.READY)]
    assert events[0] == (
        "init",
        {
            "init_info": {
                "model_name": "policy",
                "server_url": "dns:///mx:50051",
                "initial_base_version_id": "base-uid",
                "launch_checkpoint": "/models/model",
                "preparation_cache_dir": "/mxdelta/mxprep",
                "object_storage_type": "S3",
                "object_storage_endpoint_url": "http://minio:9000",
                "object_storage_region_name": "us-west-2",
                "registration_ttl_seconds": None,
                "lease_ttl_seconds": None,
                "max_transfer_attempts": 4,
                "rpc_timeout_seconds": 321.0,
            }
        },
    )
    assert [event for event, _payload in events[1:]] == [
        "pause",
        "flush",
        "start",
        "update:uid-1",
        "finish:1",
        "continue",
    ]
    assert instance.weight_version == 1
    assert instance._current_version_id == "uid-1"
    assert instance.pop_metrics() == {
        "perf/update_weights_density": 0.25,
        "perf/update_weights_wire_bytes": 246,
        "perf/mx_stage_delta_time": 17.0,
        "perf/mx_publish_object_storage_time": 18.0,
        "perf/mx_control_create_weight_version": 11.0,
        "perf/mx_stage_shard": 12.0,
        "perf/mx_publish_shard": 13.0,
        "perf/mx_publish_server": 14.0,
        "perf/mx_update_activate_time": 15.0,
    }
    assert reductions == [
        ([25, 100, 123], torch.distributed.ReduceOp.SUM),
        (
            [7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            torch.distributed.ReduceOp.MAX,
        ),
    ]


def test_reconnecting_the_same_vllm_cohort_is_a_noop():
    instance = updater(FakeControl(), FakeTrainer())
    events = []
    engine = FakeEngine(events)

    instance.connect_rollout_engines([engine], object())
    instance.connect_rollout_engines([engine], object())

    assert [event for event, _payload in events if event == "init"] == ["init"]


def test_failed_vllm_initialization_is_retried_for_the_same_cohort():
    instance = updater(FakeControl(), FakeTrainer())
    events = []
    engine = FakeEngine(events, init_error="init failed")

    with pytest.raises(ModelExpressUpdateError, match="init failed"):
        instance.connect_rollout_engines([engine], object())

    assert not instance.is_rollout_engines_fresh()
    engine.init_error = None
    instance.connect_rollout_engines([engine], object())

    assert instance.is_rollout_engines_fresh()
    assert [event for event, _payload in events if event == "init"] == ["init", "init"]


def test_ready_transition_retries_without_republishing():
    control = FakeControl()
    control.state_error = "ready failed"
    trainer = FakeTrainer()
    instance = updater(control, trainer)
    engine = FakeEngine([])
    instance.connect_rollout_engines([engine], object())
    instance.update_weights()

    with pytest.raises(ModelExpressUpdateError, match="ready failed"):
        instance.update_weights()

    assert instance._pending_version_id == "uid-1"
    assert instance._pending_published
    assert not instance._pending_ready
    assert len(control.created) == 1
    assert len(trainer.publishes) == 1

    instance.update_weights()

    assert instance.weight_version == 1
    assert len(control.created) == 1
    assert len(trainer.publishes) == 1


def test_failed_vllm_update_retries_the_ready_version_without_republishing():
    control = FakeControl()
    trainer = FakeTrainer()
    instance = updater(control, trainer)
    events = []
    engine = FakeEngine(events, update_error="install failed")
    instance.connect_rollout_engines([engine], object())
    instance.update_weights()

    with pytest.raises(ModelExpressUpdateError, match="install failed"):
        instance.update_weights()

    assert instance.weight_version == 0
    assert instance._pending_version_id == "uid-1"
    assert len(control.created) == 1
    assert len(trainer.publishes) == 1
    assert not any(event == "continue" for event, _payload in events[1:])

    engine.update_error = None
    instance.update_weights()

    assert instance.weight_version == 1
    assert instance._current_version_id == "uid-1"
    assert instance._pending_version_id is None
    assert len(control.created) == 1
    assert len(trainer.publishes) == 1


def test_multi_engine_activation_runs_in_bulk_phases():
    instance = updater(FakeControl(), FakeTrainer())
    events = []
    first = FakeEngine(events)
    second = FakeEngine(events)
    instance.connect_rollout_engines([first, second], object())
    events.clear()

    instance._activate_on_rank_zero("uid-1", 1)

    assert [event for event, _payload in events] == [
        "pause",
        "pause",
        "flush",
        "flush",
        "start",
        "start",
        "update:uid-1",
        "update:uid-1",
        "finish:1",
        "finish:1",
        "continue",
        "continue",
    ]
