from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle

from vime.utils.disk_delta import make_tensor_reader
from vime.utils.distributed_utils import get_gloo_group

from .update_weight_from_disk_delta import UpdateWeightFromDiskDelta


class ModelExpressUpdateError(RuntimeError):
    pass


class UpdateWeightFromModelExpress(UpdateWeightFromDiskDelta):
    """Vime publisher and vLLM weight-transfer lifecycle integration."""

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        publisher=None,
    ) -> None:
        del weights_getter
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self._config = dict(args.modelexpress_config)
        self.weight_version = int(self._config.get("initial_version", "0"))
        self.rollout_engines: Sequence[ActorHandle] | None = None
        self._connection_stale = False
        self._baseline_captured = False
        self._metrics: dict[str, float] = {}

        if publisher is None:
            from modelexpress.refit import Publisher, PublisherConfig, S3Config

            publisher = Publisher(
                launch_checkpoint=vars(args)["hf_checkpoint"],
                bucket_bytes=args.update_weight_buffer_size,
                group=get_gloo_group(),
            )
            publisher.initialize(
                PublisherConfig(
                    model_id=self._config["model_id"],
                    catalog_endpoint=self._config["catalog_endpoint"],
                    s3=S3Config(
                        bucket=self._config["s3_bucket"],
                        prefix=self._config.get("s3_prefix", ""),
                        endpoint_url=self._config.get("s3_endpoint"),
                    ),
                )
            )
        self._publisher = publisher
        self._catalog = publisher.catalog
        self._control = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="modelexpress-control")
            if dist.get_rank() == 0
            else None
        )
        self._publisher.publish_version("0")

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
        self._connection_stale = False
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        version = str(self.weight_version)
        launch_pending = self._publisher.pending_version == version
        if dist.get_rank() == 0:
            init_info = {
                "model_id": self._config["model_id"],
                "catalog_endpoint": self._config["catalog_endpoint"],
                "initial_version": version,
                "preparation_cache_dir": self._config["preparation_cache_dir"],
                "ready_timeout_seconds": float(self._config.get("ready_timeout_seconds", 600.0)),
                "s3_endpoint_url": self._config.get("s3_endpoint"),
            }
            try:
                ray.get(
                    [
                        engine.init_weight_transfer_engine.remote({"init_info": init_info})
                        for engine in self.rollout_engines
                    ]
                )
            except Exception as error:
                raise ModelExpressUpdateError(f"vLLM ModelExpress initialization failed: {error}") from error
            if launch_pending:
                self._catalog.commit_revision(self._config["model_id"], version)
        if launch_pending:
            self._publisher.wait_for_commit(version)

    def disconnect_rollout_engines(self) -> None:
        self.rollout_engines = None
        self._connection_stale = True

    def pop_metrics(self) -> dict[str, float]:
        metrics, self._metrics = self._metrics, {}
        return metrics

    @torch.no_grad()
    def update_weights(self) -> None:
        if not self._baseline_captured:
            self._publisher.capture_baseline(
                self._for_each_hf_bucket,
                make_tensor_reader(vars(self.args)["hf_checkpoint"]),
            )
            self._baseline_captured = True
            return
        if self.rollout_engines is None:
            raise ModelExpressUpdateError("rollout engines are not connected")

        target_version = str(self.weight_version + 1)
        self._publisher.publish_version(
            target_version,
            base_version=str(self.weight_version),
            gather_hf_buckets=self._for_each_hf_bucket,
        )
        future = None
        if dist.get_rank() == 0:
            assert self._control is not None
            future = self._control.submit(self._activate_on_rank_zero, target_version)
        try:
            self._publisher.wait_for_commit(target_version, future)
        except Exception as error:
            if isinstance(error, ModelExpressUpdateError):
                raise
            raise ModelExpressUpdateError(f"ModelExpress update failed for {target_version}: {error}") from error
        dist.barrier(group=get_gloo_group())
        self.weight_version += 1
        self._metrics = self._publisher.pop_metrics()

    def _for_each_hf_bucket(self, consume: Callable[[Any], None]) -> None:
        for chunk_iter in (
            self._iter_non_expert_chunks(),
            self._iter_expert_chunks(),
        ):
            for bucket in chunk_iter:
                consume(bucket)
            dist.barrier(group=get_gloo_group())

    def _activate_on_rank_zero(self, target_version: str) -> None:
        engines = tuple(self.rollout_engines or ())
        if not engines:
            raise ModelExpressUpdateError("ModelExpress requires rollout engines")
        try:
            ray.get([engine.pause_generation.remote() for engine in engines])
            ray.get([engine.flush_cache.remote() for engine in engines])
            ray.get([engine.start_weight_update.remote() for engine in engines])
            ray.get([engine.update_weights_from_modelexpress.remote(target_version) for engine in engines])
            ray.get([engine.finish_weight_update.remote() for engine in engines])
        except Exception as error:
            raise ModelExpressUpdateError(f"vLLM ModelExpress install failed for {target_version}: {error}") from error
        self._catalog.commit_revision(self._config["model_id"], target_version)
        ray.get([engine.continue_generation.remote() for engine in engines])
