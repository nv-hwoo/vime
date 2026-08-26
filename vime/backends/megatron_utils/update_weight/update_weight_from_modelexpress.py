from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle

from vime.utils.distributed_utils import get_gloo_group

from .update_weight_from_disk_delta import UpdateWeightFromDiskDelta


class ModelExpressUpdateError(RuntimeError):
    pass


_UPDATE_PHASE_METRICS = (
    "perf/mx_control_create_weight_version",
    "perf/mx_stage_shard",
    "perf/mx_publish_shard",
    "perf/mx_publish_server",
    "perf/mx_update_activate_time",
)


class UpdateWeightFromModelExpress(UpdateWeightFromDiskDelta):
    """Publish canonical S3 deltas and update vLLM through ModelExpress."""

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        control_client=None,
        trainer_client=None,
    ) -> None:
        del weights_getter
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self._config = dict(args.modelexpress_config)
        self.weight_version = int(self._config.get("initial_version", 0))
        self._current_version_id = str(self._config["initial_base_version_id"])
        self._pending_version_id: str | None = None
        self._pending_published = False
        self._pending_ready = False
        self.rollout_engines: Sequence[ActorHandle] | None = None
        self._connection_stale = False
        self._baseline_captured = False
        self._metrics: dict[str, int | float] = {}

        from modelexpress_rl import (
            ModelExpressControlClient,
            ModelExpressTrainerClient,
            ModelExpressTrainerConfig,
            ObjectStorageConfig,
            ObjectStorageSource,
            ObjectStorageType,
            TrainerStagingMode,
            WeightPayloadFormat,
            WeightVersionRef,
            WeightVersionState,
        )

        self._WeightPayloadFormat = WeightPayloadFormat
        self._ObjectStorageSource = ObjectStorageSource
        self._WeightVersionRef = WeightVersionRef
        self._WeightVersionState = WeightVersionState
        rpc_timeout_seconds = float(self._config.get("rpc_timeout_seconds", 30.0))
        self._object_storage_config = ObjectStorageConfig(
            storage_type=ObjectStorageType.S3,
            uri_prefix=self._config["s3_uri_prefix"],
            initial_base_version_id=self._current_version_id,
            launch_checkpoint=vars(args)["hf_checkpoint"],
            endpoint_url=self._config.get("s3_endpoint_url"),
            region_name=self._config.get("s3_region_name"),
        )

        if trainer_client is None:
            registration_ttl = self._config.get("registration_ttl_seconds")
            trainer_client = ModelExpressTrainerClient.initialize(
                ModelExpressTrainerConfig(
                    model_name=self._config["model_name"],
                    staging_mode=TrainerStagingMode.WRITE_TO_STORAGE,
                    payload_format=WeightPayloadFormat.XOR_DELTA,
                    server_url=self._config["server_url"],
                    registration_ttl_seconds=(int(registration_ttl) if registration_ttl is not None else None),
                    rpc_timeout_seconds=rpc_timeout_seconds,
                    process_group=get_gloo_group(),
                    object_storage=self._object_storage_config,
                )
            )
        self._trainer = trainer_client

        if dist.get_rank() == 0:
            self._control = control_client or ModelExpressControlClient.connect(
                server_url=self._config["server_url"],
                rpc_timeout_seconds=rpc_timeout_seconds,
            )
        else:
            self._control = None

    def is_rollout_engines_fresh(self) -> bool:
        return self.rollout_engines is not None and not self._connection_stale

    def mark_engine_connection_stale(self) -> None:
        self._connection_stale = True

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        engine_parallel_configs: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        del rollout_engine_lock, engine_gpu_counts, engine_gpu_offsets, engine_parallel_configs
        connected = tuple(rollout_engines)
        if self.rollout_engines == connected and not self._connection_stale:
            return
        self.rollout_engines = connected
        self._connection_stale = True
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )

        registration_ttl = self._config.get("registration_ttl_seconds")
        lease_ttl = self._config.get("lease_ttl_seconds")

        init_info = {
            "model_name": self._config["model_name"],
            "server_url": self._config["server_url"],
            "initial_base_version_id": self._current_version_id,
            "launch_checkpoint": vars(self.args)["hf_checkpoint"],
            "preparation_cache_dir": self._config["preparation_cache_dir"],
            "object_storage_type": self._object_storage_config.storage_type.value,
            "object_storage_endpoint_url": self._config.get("s3_endpoint_url"),
            "object_storage_region_name": self._config.get("s3_region_name"),
            "registration_ttl_seconds": int(registration_ttl) if registration_ttl is not None else None,
            "lease_ttl_seconds": int(lease_ttl) if lease_ttl is not None else None,
            "max_transfer_attempts": int(self._config.get("max_transfer_attempts", 3)),
            "rpc_timeout_seconds": float(self._config.get("rpc_timeout_seconds", 30.0)),
        }

        def initialize_engines():
            ray.get([engine.init_weight_transfer_engine.remote({"init_info": init_info}) for engine in connected])

        self._rank_zero_call(initialize_engines, "vLLM ModelExpress initialization failed")
        self._connection_stale = False

    def disconnect_rollout_engines(self) -> None:
        self.rollout_engines = None
        self._connection_stale = True

    def pop_metrics(self) -> dict[str, int | float]:
        metrics, self._metrics = self._metrics, {}
        return metrics

    def _rank_zero_call(self, action: Callable[[], Any], description: str) -> Any:
        result = [None, None]
        if dist.get_rank() == 0:
            try:
                result[0] = action()
            except Exception as error:
                result[1] = str(error)
        dist.broadcast_object_list(result, src=0, group=get_gloo_group())
        if result[1] is not None:
            raise ModelExpressUpdateError(f"{description}: {result[1]}")
        return result[0]

    @torch.no_grad()
    def update_weights(self) -> None:
        if not self._baseline_captured:
            self._trainer.prepare_delta_base(hf_tensor_iter=self._iter_hf_buckets())
            self._baseline_captured = True
            return
        if self.rollout_engines is None:
            raise ModelExpressUpdateError("rollout engines are not connected")

        phase_times = dict.fromkeys(_UPDATE_PHASE_METRICS, 0.0)
        target_version_number = self.weight_version + 1
        if self._pending_version_id is None:
            if dist.get_rank() == 0:
                assert self._control is not None
            phase_started = perf_counter()
            target_version_id = self._rank_zero_call(
                lambda: (
                    self._control.create_weight_version(
                        model_name=self._config["model_name"],
                        version_number=target_version_number,
                        idempotency_key=(f"vime:{self._current_version_id}:v{target_version_number}"),
                        payload_format=self._WeightPayloadFormat.XOR_DELTA,
                        base_version_id=self._current_version_id,
                        object_storage=self._ObjectStorageSource(
                            storage_type=self._object_storage_config.storage_type,
                            uri=self._object_storage_config.root_uri(target_version_number),
                        ),
                        state=self._WeightVersionState.STAGING,
                    ).version_id
                ),
                f"ModelExpress version {target_version_number} creation failed",
            )
            phase_times["perf/mx_control_create_weight_version"] = perf_counter() - phase_started
            self._pending_version_id = str(target_version_id)

        assert self._pending_version_id is not None
        version = self._WeightVersionRef(self._pending_version_id)
        if not self._pending_published:
            phase_started = perf_counter()
            staged = self._trainer.stage_shard(
                version=version,
                hf_tensor_iter=self._iter_hf_buckets(),
            )
            phase_times["perf/mx_stage_shard"] = perf_counter() - phase_started

            phase_started = perf_counter()
            staged.publish()
            phase_times["perf/mx_publish_shard"] = perf_counter() - phase_started
            self._pending_published = True

        if not self._pending_ready:
            dist.barrier(group=get_gloo_group())
            phase_started = perf_counter()
            self._rank_zero_call(
                lambda: self._control.update_weight_version_state(
                    version.version_id,
                    self._WeightVersionState.READY,
                ),
                f"ModelExpress version {target_version_number} activation failed",
            )
            phase_times["perf/mx_publish_server"] = perf_counter() - phase_started
            self._pending_ready = True

        target_version_id = self._pending_version_id
        assert target_version_id is not None
        phase_started = perf_counter()
        self._rank_zero_call(
            lambda: self._activate_on_rank_zero(
                target_version_id,
                target_version_number,
            ),
            f"ModelExpress version {target_version_number} install failed",
        )
        phase_times["perf/mx_update_activate_time"] = perf_counter() - phase_started
        self._current_version_id = target_version_id
        self._pending_version_id = None
        self._pending_published = False
        self._pending_ready = False
        self.weight_version = target_version_number
        self._metrics = self._gather_metrics(
            phase_times=phase_times,
            group=get_gloo_group(),
        )

    def _iter_hf_buckets(self):
        for chunk_iter in (
            self._iter_non_expert_chunks(),
            self._iter_expert_chunks(),
        ):
            yield from chunk_iter
            dist.barrier(group=get_gloo_group())

    def _gather_metrics(
        self,
        *,
        phase_times: Mapping[str, float],
        group: Any,
    ) -> dict[str, int | float]:
        local_metrics = self._trainer.pop_metrics()
        counts = torch.tensor(
            [
                local_metrics.get("changed_bytes", 0),
                local_metrics.get("total_bytes", 0),
                local_metrics.get("wire_bytes", 0),
            ],
            dtype=torch.int64,
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)

        timings = torch.tensor(
            [
                local_metrics.get("stage_delta_time", 0.0),
                local_metrics.get("publish_object_storage_time", 0.0),
                *(phase_times[name] for name in _UPDATE_PHASE_METRICS),
            ],
            dtype=torch.float64,
        )
        dist.all_reduce(timings, op=dist.ReduceOp.MAX, group=group)

        changed_bytes, total_bytes, wire_bytes = counts.tolist()
        stage_delta_time, publish_object_storage_time, *phase_values = timings.tolist()
        return {
            "perf/update_weights_density": changed_bytes / max(total_bytes, 1),
            "perf/update_weights_wire_bytes": wire_bytes,
            "perf/mx_stage_delta_time": stage_delta_time,
            "perf/mx_publish_object_storage_time": publish_object_storage_time,
            **dict(zip(_UPDATE_PHASE_METRICS, phase_values, strict=True)),
        }

    def _activate_on_rank_zero(
        self,
        target_version_id: str,
        target_version_number: int,
    ) -> None:
        engines = tuple(self.rollout_engines or ())
        if not engines:
            raise ModelExpressUpdateError("ModelExpress requires rollout engines")
        ray.get([engine.pause_generation.remote() for engine in engines])
        ray.get([engine.flush_cache.remote() for engine in engines])
        ray.get([engine.start_weight_update.remote() for engine in engines])
        ray.get([engine.update_weights_from_modelexpress.remote(target_version_id) for engine in engines])
        ray.get([engine.finish_weight_update.remote(str(target_version_number)) for engine in engines])
        ray.get([engine.continue_generation.remote() for engine in engines])
