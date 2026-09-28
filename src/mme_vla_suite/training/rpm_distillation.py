"""Stage-A helpers for distilling RPM reads into the local compressor."""

from collections.abc import Mapping, Sequence
import dataclasses

import flax.nnx as nnx
import flax.nnx.bridge.variables as nnx_bridge_variables
import jax
import jax.numpy as jnp
import numpy as np
import optax

from mme_vla_suite.models.integration.history_gemma import MemoryAttention
from mme_vla_suite.models.representation.percep_mem import LocalCompressor
from mme_vla_suite.models.representation.percep_mem import pack_encoded_frames
from mme_vla_suite.shared.rpm_sampling import GROUPS
from mme_vla_suite.shared.rpm_sampling import make_plan
import openpi.training.utils as training_utils

_WARM_PLAN_TABLE = {
    name: np.stack(
        [make_plan(np.arange(count), count - 1, mode="warm")[name] for count in range(33)]
    )
    for name in ("mem_gather", "mem_mask", "mem_mass", "mem_kpos", "mem_qoffset")
}


def make_warm_metadata(batch_size: int, num_frames: int = 32):
    """Build the student layout for the exact frame order captured by the teacher."""
    if not 0 <= num_frames <= 32:
        raise ValueError("Stage-A warm distillation supports at most 32 teacher frames")
    return {
        name: jnp.broadcast_to(
            jnp.asarray(values[num_frames]),
            (batch_size, *np.shape(values[num_frames])),
        )
        for name, values in _WARM_PLAN_TABLE.items()
    }


def make_warm_metadata_from_frame_valid(frame_valid):
    """Select a warm plan per sample, preserving baseline right-padding."""
    if frame_valid.ndim != 2 or frame_valid.shape[1] != 32:
        raise ValueError("Expected baseline frame validity with shape [B, 32]")
    counts = frame_valid.astype(jnp.int32).sum(axis=1)
    return {
        name: jnp.take(jnp.asarray(values), counts, axis=0)
        for name, values in _WARM_PLAN_TABLE.items()
    }


def pack_warm_memory(compressor, encoded_frames, metadata):
    """Compress stopped teacher encodings; only ``compressor`` is trainable."""
    if encoded_frames.ndim != 4 or encoded_frames.shape[1:3] != (32, 16):
        raise ValueError("Expected teacher encodings with shape [B, 32, 16, D]")
    encoded_frames = jax.lax.stop_gradient(encoded_frames)
    encoded_frames = jnp.pad(encoded_frames, ((0, 0), (0, 24), (0, 0), (0, 0)))
    return pack_encoded_frames(
        encoded_frames,
        compressor,
        metadata["mem_gather"],
        metadata["mem_mask"],
    )


def _find_memory_attention_params(params):
    matches = []

    def visit(node):
        if not isinstance(node, Mapping):
            return
        if "mem_attn" in node:
            matches.append(node["mem_attn"])
        for value in node.values():
            visit(value)

    visit(params)
    if len(matches) != 1:
        raise ValueError(f"Expected one scanned mem_attn parameter subtree, got {len(matches)}")
    return matches[0]


def extract_frozen_memory_readers(teacher_llm, layer_indices: Sequence[int]):
    """Extract individual layers from the scanned teacher MemoryAttention params."""
    nnx_attrs = {
        name: getattr(teacher_llm, name)
        for name in teacher_llm.linen_attributes
    }
    variables = nnx_bridge_variables.nnx_attrs_to_linen_vars(nnx_attrs)
    scanned_params = _find_memory_attention_params(variables["params"])

    leaves = jax.tree.leaves(scanned_params)
    if not leaves:
        raise ValueError("Teacher mem_attn parameter subtree is empty")
    depth = leaves[0].shape[0]
    if any(leaf.shape[0] != depth for leaf in leaves):
        raise ValueError("Teacher mem_attn parameters do not share a scan layer axis")

    readers = []
    for layer_index in layer_indices:
        if not 0 <= layer_index < depth:
            raise ValueError(f"MemoryAttention layer {layer_index} exceeds depth {depth}")
        params = jax.tree.map(
            lambda value, index=layer_index: jax.lax.stop_gradient(value[index]),
            scanned_params,
        )
        readers.append({"params": params})
    return tuple(readers)


def apply_frozen_memory_readers(
    reader_variables,
    query_inputs,
    memory,
    metadata,
):
    """Read student memory with stopped teacher queries and reader parameters."""
    if len(reader_variables) != query_inputs.shape[0]:
        raise ValueError("Reader count must match the captured query layer count")
    queries = jax.lax.stop_gradient(query_inputs)
    return jnp.stack(
        [
            MemoryAttention().apply(
                jax.tree.map(jax.lax.stop_gradient, variables),
                queries[layer],
                memory,
                metadata["mem_mask"],
                metadata["mem_mass"],
                metadata["mem_kpos"],
                metadata["mem_qoffset"],
            )
            for layer, variables in enumerate(reader_variables)
        ]
    )


def read_distill_loss(student_reads, teacher_reads, query_valid):
    """Teacher-energy-normalized MSE over valid action queries."""
    target = jax.lax.stop_gradient(teacher_reads.astype(jnp.float32))
    error = (student_reads.astype(jnp.float32) - target) ** 2
    weights = query_valid.astype(jnp.float32)[None, :, :, None]
    denominator = jnp.maximum(weights.sum() * target.shape[-1], 1.0)
    mse = (error * weights).sum(axis=(1, 2, 3)) / denominator
    energy = (target**2 * weights).sum(axis=(1, 2, 3)) / denominator
    return (mse / jnp.maximum(energy, 1e-4)).mean()


def teacher_read_energy(teacher_reads, query_valid):
    """Mean teacher energy using the same valid-query normalization as the loss."""
    target = jax.lax.stop_gradient(teacher_reads.astype(jnp.float32))
    weights = query_valid.astype(jnp.float32)[None, :, :, None]
    denominator = jnp.maximum(weights.sum() * target.shape[-1], 1.0)
    return ((target**2 * weights).sum(axis=(1, 2, 3)) / denominator).mean()


def compressor_distill_loss(
    compressor,
    encoded_frames,
    query_inputs,
    teacher_reads,
    query_valid,
    reader_variables,
    metadata=None,
):
    """Stage-A objective whose sole differentiable module argument is compressor."""
    if metadata is None:
        metadata = make_warm_metadata(encoded_frames.shape[0])
    memory = pack_warm_memory(compressor, encoded_frames, metadata)
    student_reads = apply_frozen_memory_readers(
        reader_variables, query_inputs, memory, metadata
    )
    return read_distill_loss(student_reads, teacher_reads, query_valid)


def init_compressor_train_state(compressor: LocalCompressor, tx):
    """Create an optimizer state containing compressor parameters and nothing else."""
    params = nnx.state(compressor)
    return training_utils.TrainState(
        step=0,
        params=params,
        model_def=nnx.graphdef(compressor),
        tx=tx,
        opt_state=tx.init(params),
        ema_decay=None,
        ema_params=None,
    )


def _mean_pool_compressor(encoded_frames):
    cells = jnp.take(encoded_frames, jnp.asarray(GROUPS), axis=-2)
    return cells.mean(axis=-2)


def compressor_train_step(state, captured, reader_variables, metadata=None):
    """Update only the compressor from one stopped teacher capture."""
    compressor = nnx.merge(state.model_def, state.params)
    if metadata is None:
        metadata = (
            make_warm_metadata_from_frame_valid(captured["frame_valid"])
            if "frame_valid" in captured
            else make_warm_metadata(captured["encoded_frames"].shape[0])
        )

    def loss_fn(module):
        memory = pack_warm_memory(module, captured["encoded_frames"], metadata)
        student_reads = apply_frozen_memory_readers(
            reader_variables, captured["query_inputs"], memory, metadata
        )
        loss = read_distill_loss(
            student_reads, captured["teacher_reads"], captured["query_valid"]
        )

        pooling_memory = pack_warm_memory(
            _mean_pool_compressor, captured["encoded_frames"], metadata
        )
        pooling_reads = apply_frozen_memory_readers(
            reader_variables,
            captured["query_inputs"],
            pooling_memory,
            metadata,
        )
        metrics = {
            "loss": loss,
            "pooling_relative_mse": read_distill_loss(
                pooling_reads,
                captured["teacher_reads"],
                captured["query_valid"],
            ),
            "teacher_energy": teacher_read_energy(
                captured["teacher_reads"], captured["query_valid"]
            ),
        }
        return loss, metrics

    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(compressor)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    new_state = dataclasses.replace(
        state,
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )
    metrics = {
        **metrics,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(new_params),
    }
    return new_state, metrics
