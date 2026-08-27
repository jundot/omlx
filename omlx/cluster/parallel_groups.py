# SPDX-License-Identifier: Apache-2.0
"""Two-dimensional MLX process groups for TP x pipeline execution.

The signed rank convention is ``rank = stage * tp_size + tp_rank``.  MLX's
global group remains the control/scheduling group, while model math uses:

* one tensor group for all ranks in the same pipeline stage; and
* one pipeline group for equal tensor ranks across every stage.

Keeping this construction in one module prevents the loader, forward path and
telemetry from independently inventing incompatible rank maps.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParallelGroups:
    world: Any
    tensor: Any | None
    pipeline: Any | None
    world_size: int
    rank: int
    tensor_parallel_size: int
    pipeline_stages: int
    tensor_rank: int
    pipeline_stage: int

    @property
    def hybrid(self) -> bool:
        return self.tensor_parallel_size > 1 and self.pipeline_stages > 1


def hybrid_group_split_supported(
    backend: str,
    *,
    mx_module: Any | None = None,
) -> bool:
    """Whether the loaded MLX backend can execute orthogonal subgroups."""

    selected = "jaccl" if str(backend).startswith("jaccl") else str(backend)
    if mx_module is None:
        try:
            import mlx.core as mx_module
        except ImportError:
            return False
    capability = getattr(
        getattr(mx_module, "distributed", None),
        "is_group_split_available",
        None,
    )
    if not callable(capability):
        return False
    try:
        return bool(capability(selected))
    except (RuntimeError, TypeError, ValueError):
        return False


def split_parallel_groups(world: Any, tensor_parallel_size: int) -> ParallelGroups:
    """Split an MLX world into the exact groups named by the signed plan."""

    world_size = int(world.size())
    rank = int(world.rank())
    if not 1 <= tensor_parallel_size <= world_size:
        raise ValueError("tensor_parallel_size must be within the MLX world")
    if world_size % tensor_parallel_size:
        raise ValueError("MLX world size must be divisible by tensor_parallel_size")

    pipeline_stages = world_size // tensor_parallel_size
    pipeline_stage = rank // tensor_parallel_size
    tensor_rank = rank % tensor_parallel_size

    if tensor_parallel_size == 1:
        tensor_group = None
        pipeline_group = world if pipeline_stages > 1 else None
    elif pipeline_stages == 1:
        tensor_group = world
        pipeline_group = None
    else:
        # Every process calls split in the same order. ``color`` chooses the
        # subgroup and ``key`` preserves the signed local rank convention.
        tensor_group = world.split(pipeline_stage, tensor_rank)
        pipeline_group = world.split(tensor_rank, pipeline_stage)
        if int(tensor_group.size()) != tensor_parallel_size:
            raise RuntimeError("MLX tensor subgroup size does not match the plan")
        if int(tensor_group.rank()) != tensor_rank:
            raise RuntimeError("MLX tensor subgroup rank does not match the plan")
        if int(pipeline_group.size()) != pipeline_stages:
            raise RuntimeError("MLX pipeline subgroup size does not match the plan")
        if int(pipeline_group.rank()) != pipeline_stage:
            raise RuntimeError("MLX pipeline subgroup rank does not match the plan")

    return ParallelGroups(
        world=world,
        tensor=tensor_group,
        pipeline=pipeline_group,
        world_size=world_size,
        rank=rank,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_stages=pipeline_stages,
        tensor_rank=tensor_rank,
        pipeline_stage=pipeline_stage,
    )


_ACTIVE_PIPELINE_GROUP: ContextVar[Any | None] = ContextVar(
    "omlx_active_pipeline_group",
    default=None,
)


@contextmanager
def install_pipeline_group_routing(
    model: Any,
    pipeline_group: Any | None,
    *,
    mx_module: Any,
) -> Iterator[None]:
    """Route group-less pipeline collectives through a pipeline subgroup.

    Pinned MLX-LM pipeline models call ``send``, ``recv_like`` and the final
    ``all_gather`` without a group argument. That is correct for pure pipeline
    execution, where the pipeline group is the world, but deadlocks a hybrid
    graph because tensor collectives must use the orthogonal subgroup.

    The route is active only while the concrete pipeline model's ``__call__``
    executes. Server coordination outside the model therefore continues to use
    the global world, and tensor modules that pass their group explicitly are
    untouched.
    """

    if pipeline_group is None or int(pipeline_group.size()) <= 1:
        yield
        return

    pipeline_model = getattr(model, "model", None)
    if pipeline_model is None:
        raise RuntimeError("hybrid model does not expose its pipeline module")
    model_class = type(pipeline_model)
    original_call = model_class.__call__
    previous_group = getattr(pipeline_model, "_omlx_pipeline_group", None)
    distributed = mx_module.distributed
    originals: dict[str, Any] = {}

    def routed_call(instance: Any, *args: Any, **kwargs: Any) -> Any:
        group = getattr(instance, "_omlx_pipeline_group", None)
        token = _ACTIVE_PIPELINE_GROUP.set(group)
        try:
            return original_call(instance, *args, **kwargs)
        finally:
            _ACTIVE_PIPELINE_GROUP.reset(token)

    def wrap_collective(function: Any):
        def routed(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get("group") is None:
                group = _ACTIVE_PIPELINE_GROUP.get()
                if group is not None:
                    kwargs["group"] = group
            return function(*args, **kwargs)

        return routed

    try:
        pipeline_model._omlx_pipeline_group = pipeline_group
        model_class.__call__ = routed_call
        for name in ("send", "recv", "recv_like", "all_gather"):
            function = getattr(distributed, name, None)
            if callable(function):
                originals[name] = function
                setattr(distributed, name, wrap_collective(function))
        yield
    finally:
        for name, function in originals.items():
            setattr(distributed, name, function)
        model_class.__call__ = original_call
        if previous_group is None:
            with suppress(AttributeError):
                delattr(pipeline_model, "_omlx_pipeline_group")
        else:
            pipeline_model._omlx_pipeline_group = previous_group


__all__ = [
    "ParallelGroups",
    "hybrid_group_split_supported",
    "install_pipeline_group_routing",
    "split_parallel_groups",
]
