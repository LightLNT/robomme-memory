import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax
import pytest

from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.integration.history_gemma import HistoryBlock
from mme_vla_suite.models.integration.history_gemma import MemoryAttention
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
from mme_vla_suite.models.integration.history_pi0 import _apply_with_read_capture
from mme_vla_suite.models.integration.history_pi0 import _read_capture_value
from mme_vla_suite.models.representation.percep_mem import LocalCompressor
from mme_vla_suite.models.representation.percep_mem import PerceptualMemory
from mme_vla_suite.shared.rpm_sampling import GROUPS
from mme_vla_suite.shared.rpm_sampling import make_plan
from mme_vla_suite.training.rpm_distillation import apply_frozen_memory_readers
from mme_vla_suite.training.rpm_distillation import compressor_distill_loss
from mme_vla_suite.training.rpm_distillation import compressor_train_step
from mme_vla_suite.training.rpm_distillation import extract_frozen_memory_readers
from mme_vla_suite.training.rpm_distillation import init_compressor_train_state
from mme_vla_suite.training.rpm_distillation import make_warm_metadata
from mme_vla_suite.training.rpm_distillation import make_warm_metadata_from_frame_valid
from mme_vla_suite.training.rpm_distillation import pack_warm_memory
from mme_vla_suite.training.rpm_distillation import read_distill_loss
from openpi.models.gemma import Config


@pytest.mark.parametrize(
    ("length", "fine_frames", "token_count"),
    [(0, 0, 0), (32, 32, 512), (33, 31, 504), (40, 29, 508), (56, 24, 512)],
)
def test_make_plan_budget_and_layout(length, fine_frames, token_count):
    plan = make_plan(np.arange(length), length - 1)

    assert plan["frame_valid"].sum() == length
    assert plan["frame_high"].sum() == fine_frames
    assert plan["mem_mask"].sum() == token_count
    assert plan["mem_mask"].shape == (512,)
    assert plan["mem_mass"].shape == (512,)
    assert np.isclose(plan["mem_mass"][plan["mem_mask"]].sum(), 16 * length)
    assert plan["mem_qoffset"] == max(512, 16 * length)


def test_make_plan_uses_only_available_ids_before_cutoff():
    plan = make_plan([40, 2, 20, 5, 11], cutoff=11)

    np.testing.assert_array_equal(plan["frame_ids"][:3], [2, 5, 11])
    assert plan["frame_valid"].sum() == 3


def test_local_compressor_starts_as_spatial_region_mean():
    compressor = LocalCompressor(dim=8, hidden=4, rngs=nnx.Rngs(0), dtype=jnp.float32)
    frames = jax.random.normal(jax.random.key(1), (2, 3, 16, 8))

    output = compressor(frames)
    expected = jnp.take(frames, jnp.asarray(GROUPS), axis=-2).mean(axis=-2)

    np.testing.assert_allclose(output, expected, rtol=0, atol=0)


def test_perceptual_memory_packs_to_read_budget_and_zeros_padding():
    config = OmegaConf.create({
        "budget": 512,
        "token_per_image": 16,
        "memory_token_dim": 8,
        "use_pos_emb": False,
        "use_state_emb": False,
        "memory_feature": {
            "img": {"input_dim": 4},
            "pos": {"input_dim": 2, "hidden_dim": 2},
            "state": {"input_dim": 3, "hidden_dim": 3},
        },
        "perceptual_memory": {"type": "multires_frame_sampling"},
        "multires": {"max_frames": 56, "raw_capacity": 896, "compressor_hidden": 4},
    })
    memory = PerceptualMemory(config, rngs=nnx.Rngs(0))
    plan = make_plan(np.arange(33), cutoff=32)
    image = jax.random.normal(jax.random.key(2), (1, 896, 4))
    position = jnp.zeros((1, 896, 2), dtype=jnp.float32)
    state = jnp.zeros((1, 896, 3), dtype=jnp.float32)

    tokens, _, _ = memory(
        image,
        position,
        state,
        mem_gather=jnp.asarray(plan["mem_gather"])[None],
        mem_mask=jnp.asarray(plan["mem_mask"])[None],
    )

    assert tokens.shape == (1, 512, 8)
    np.testing.assert_array_equal(
        np.asarray(tokens[0, ~plan["mem_mask"]]),
        np.zeros((512 - plan["mem_mask"].sum(), 8)),
    )


def test_memory_attention_mass_matches_repeated_identical_tokens():
    attention = MemoryAttention()
    key = jax.random.key(3)
    x = jax.random.normal(key, (1, 2, 1024))
    a = jax.random.normal(jax.random.key(4), (1, 1, 1024))
    b = jax.random.normal(jax.random.key(5), (1, 1, 1024))
    compressed = jnp.concatenate([a, b], axis=1)
    compressed_mask = jnp.ones((1, 2), dtype=jnp.bool_)
    variables = attention.init(
        key,
        x,
        compressed,
        compressed_mask,
        jnp.asarray([[4.0, 1.0]]),
        jnp.asarray([[0.0, 16.0]]),
        jnp.asarray([32.0]),
    )

    compressed_output = attention.apply(
        variables,
        x,
        compressed,
        compressed_mask,
        jnp.asarray([[4.0, 1.0]]),
        jnp.asarray([[0.0, 16.0]]),
        jnp.asarray([32.0]),
    )
    expanded = jnp.concatenate([jnp.repeat(a, 4, axis=1), b], axis=1)
    expanded_output = attention.apply(
        variables,
        x,
        expanded,
        jnp.ones((1, 5), dtype=jnp.bool_),
        jnp.ones((1, 5), dtype=jnp.float32),
        jnp.asarray([[0.0, 0.0, 0.0, 0.0, 16.0]]),
        jnp.asarray([32.0]),
    )

    # Full 1024-wide output projections can accumulate backend-dependent float32
    # differences even though the corrected attention distributions are equivalent.
    np.testing.assert_allclose(compressed_output, expanded_output, rtol=5e-4, atol=5e-4)


def test_memory_attention_all_masked_is_zero():
    attention = MemoryAttention()
    key = jax.random.key(6)
    x = jax.random.normal(key, (1, 2, 1024))
    memory = jax.random.normal(jax.random.key(7), (1, 3, 1024))
    mask = jnp.zeros((1, 3), dtype=jnp.bool_)
    variables = attention.init(key, x, memory, mask)

    output = attention.apply(variables, x, memory, mask)

    assert jnp.isfinite(output).all()
    np.testing.assert_array_equal(np.asarray(output), np.zeros((1, 2, 1024)))


def test_input_specs_separate_raw_capacity_and_read_budget():
    rpm_history = get_history_config("perceptual-rpm-modul.yaml")
    rpm_config = HistoryPi0Config(use_history=True, history_config=rpm_history)
    rpm_obs, _ = rpm_config.inputs_spec(batch_size=2)

    assert rpm_obs.static_image_emb.shape == (2, 896, 2048)
    assert rpm_obs.static_mask.shape == (2, 896)
    assert rpm_obs.mem_gather.shape == (2, 512)
    assert rpm_obs.mem_mask.shape == (2, 512)
    assert rpm_obs.mem_qoffset.shape == (2,)

    baseline_history = get_history_config("perceptual-framesamp-modul.yaml")
    baseline_config = HistoryPi0Config(use_history=True, history_config=baseline_history)
    baseline_obs, _ = baseline_config.inputs_spec(batch_size=2)

    assert baseline_obs.static_image_emb.shape == (2, 512, 2048)
    assert baseline_obs.static_mask.shape == (2, 512)
    assert baseline_obs.mem_gather is None


class _ScannedHistoryBlock(nn.Module):
    config: Config

    @nn.compact
    def __call__(
        self,
        xs,
        positions,
        attn_mask,
        mem_seq,
        mem_mask,
        mem_mass,
        mem_kpos,
        mem_qoffset,
        capture_reads=False,  # noqa: FBT002 -- positional for Linen scan axes
    ):
        block_cls = nn.remat(
            HistoryBlock,
            prevent_cse=False,
            static_argnums=(10, 11),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        layers = nn.scan(
            block_cls,
            variable_axes={"params": 0, "rpm_reads": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),
            length=1,
        )(
            configs=(self.config, self.config),
            integration_type="modulation",
        )
        return layers(
            xs,
            None,
            positions,
            attn_mask,
            [None, None],
            mem_seq,
            mem_mask,
            mem_mass,
            mem_kpos,
            mem_qoffset,
            capture_reads,
            True,  # noqa: FBT003 -- positional for Linen scan axes
        )[0]


def test_scan_and_remat_accept_rpm_metadata():
    config = Config(
        width=1024,
        depth=1,
        mlp_dim=32,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
    )
    model = _ScannedHistoryBlock(config)
    x = jnp.zeros((1, 1, 1024), dtype=jnp.float32)
    memory = jnp.zeros((1, 4, 1024), dtype=jnp.float32)
    args = (
        [None, x],
        jnp.zeros((1, 1), dtype=jnp.int32),
        jnp.ones((1, 1, 1, 1), dtype=jnp.bool_),
        [None, memory],
        [None, jnp.ones((1, 4), dtype=jnp.bool_)],
        [None, jnp.ones((1, 4), dtype=jnp.float32)],
        [None, jnp.arange(4, dtype=jnp.float32)[None]],
        [None, jnp.asarray([512.0])],
    )

    variables = model.init(jax.random.key(8), *args)
    output = model.apply(variables, *args)

    assert output[-1].shape == x.shape
    assert jnp.isfinite(output[-1]).all()


def test_read_capture_has_layer_axis_and_is_disabled_by_default():
    config = Config(
        width=1024,
        depth=1,
        mlp_dim=32,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
    )
    model = _ScannedHistoryBlock(config)
    x = jnp.zeros((1, 2, 1024), dtype=jnp.float32)
    memory = jnp.zeros((1, 4, 1024), dtype=jnp.float32)
    args = (
        [None, x],
        jnp.arange(2, dtype=jnp.int32)[None],
        jnp.ones((1, 1, 2, 2), dtype=jnp.bool_),
        [None, memory],
        [None, jnp.ones((1, 4), dtype=jnp.bool_)],
        [None, jnp.ones((1, 4), dtype=jnp.float32)],
        [None, jnp.arange(4, dtype=jnp.float32)[None]],
        [None, jnp.asarray([512.0])],
    )

    variables = model.init(jax.random.key(9), *args)
    assert "rpm_reads" not in variables

    _, captured = model.apply(
        variables,
        *args,
        capture_reads=True,
        mutable=["rpm_reads"],
    )
    flat_capture = flax.traverse_util.flatten_dict(captured["rpm_reads"], sep="/")
    query_inputs = next(
        value for key, value in flat_capture.items() if key.endswith("query_inputs")
    )
    teacher_reads = next(
        value for key, value in flat_capture.items() if key.endswith("teacher_reads")
    )

    assert len(query_inputs) == len(teacher_reads) == 1
    assert query_inputs[0].shape == (1, 1, 2, 1024)
    assert teacher_reads[0].shape == (1, 1, 2, 1024)


def test_read_capture_bridge_does_not_mutate_nnx_state():
    config = Config(
        width=1024,
        depth=1,
        mlp_dim=32,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
    )
    model = nnx_bridge.ToNNX(_ScannedHistoryBlock(config), rngs=nnx.Rngs(10))
    x = jnp.zeros((1, 1, 1024), dtype=jnp.float32)
    memory = jnp.zeros((1, 4, 1024), dtype=jnp.float32)
    args = (
        [None, x],
        jnp.zeros((1, 1), dtype=jnp.int32),
        jnp.ones((1, 1, 1, 1), dtype=jnp.bool_),
        [None, memory],
        [None, jnp.ones((1, 4), dtype=jnp.bool_)],
        [None, jnp.ones((1, 4), dtype=jnp.float32)],
        [None, jnp.arange(4, dtype=jnp.float32)[None]],
        [None, jnp.asarray([512.0])],
    )
    model.lazy_init(*args)
    state_paths_before = set(nnx.state(model).flat_state())

    _, captured = _apply_with_read_capture(
        model,
        *args,
        capture_reads=True,
    )

    state_paths_after = set(nnx.state(model).flat_state())
    query_inputs = _read_capture_value(captured, "query_inputs")
    teacher_reads = _read_capture_value(captured, "teacher_reads")
    assert state_paths_after == state_paths_before
    assert query_inputs.shape == (1, 1, 1, 1024)
    assert teacher_reads.shape == (1, 1, 1, 1024)

    graphdef, model_state = nnx.split(model)

    @jax.jit
    def capture_inside_train_step(state):
        rebuilt = nnx.merge(graphdef, state)
        _, jit_captured = _apply_with_read_capture(
            rebuilt,
            *args,
            capture_reads=True,
        )
        return _read_capture_value(jit_captured, "teacher_reads")

    jitted_teacher_reads = capture_inside_train_step(model_state)
    assert jitted_teacher_reads.shape == (1, 1, 1, 1024)


def _initialized_scanned_teacher(seed=11):
    config = Config(
        width=1024,
        depth=1,
        mlp_dim=32,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
    )
    model = nnx_bridge.ToNNX(_ScannedHistoryBlock(config), rngs=nnx.Rngs(seed))
    x = jnp.zeros((1, 1, 1024), dtype=jnp.float32)
    memory = jnp.zeros((1, 4, 1024), dtype=jnp.float32)
    model.lazy_init(
        [None, x],
        jnp.zeros((1, 1), dtype=jnp.int32),
        jnp.ones((1, 1, 1, 1), dtype=jnp.bool_),
        [None, memory],
        [None, jnp.ones((1, 4), dtype=jnp.bool_)],
        [None, jnp.ones((1, 4), dtype=jnp.float32)],
        [None, jnp.arange(4, dtype=jnp.float32)[None]],
        [None, jnp.asarray([512.0])],
    )
    return model


def test_warm_metadata_uses_24_fine_and_8_coarse_frames():
    metadata = make_warm_metadata(batch_size=2)

    assert metadata["mem_mask"].shape == (2, 512)
    assert metadata["mem_mask"][0].sum() == 416
    assert (metadata["mem_mass"][0, :416] == 1).sum() == 24 * 16
    assert (metadata["mem_mass"][0, :416] == 4).sum() == 8 * 4
    np.testing.assert_array_equal(metadata["mem_qoffset"], [512, 512])


def test_warm_metadata_preserves_per_sample_baseline_padding():
    frame_valid = jnp.asarray(
        [
            [True] * 5 + [False] * 27,
            [True] * 32,
        ]
    )

    metadata = make_warm_metadata_from_frame_valid(frame_valid)

    assert metadata["mem_mask"][0].sum() == 5 * 16
    assert metadata["mem_mask"][1].sum() == 416
    assert metadata["mem_qoffset"][0] == metadata["mem_qoffset"][1] == 512


def test_warm_pack_uses_stopped_teacher_frames_and_compressor():
    compressor = LocalCompressor(
        dim=8, hidden=4, rngs=nnx.Rngs(12), dtype=jnp.float32
    )
    encoded = jax.random.normal(jax.random.key(13), (1, 32, 16, 8))
    metadata = make_warm_metadata(batch_size=1)

    memory = pack_warm_memory(compressor, encoded, metadata)

    assert memory.shape == (1, 512, 8)
    np.testing.assert_array_equal(
        np.asarray(memory[0, 416:]), np.zeros((96, 8), dtype=np.float32)
    )


def test_extracted_reader_matches_original_memory_attention():
    teacher = _initialized_scanned_teacher()
    reader_variables = extract_frozen_memory_readers(teacher, (0,))
    query = jax.random.normal(jax.random.key(14), (1, 1, 1, 1024))
    memory = jax.random.normal(jax.random.key(15), (1, 512, 1024))
    metadata = make_warm_metadata(batch_size=1)

    actual = apply_frozen_memory_readers(
        reader_variables, query, memory, metadata
    )[0]
    expected = MemoryAttention().apply(
        reader_variables[0],
        query[0],
        memory,
        metadata["mem_mask"],
        metadata["mem_mass"],
        metadata["mem_kpos"],
        metadata["mem_qoffset"],
    )

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_distill_loss_stops_teacher_and_only_differentiates_compressor():
    teacher = _initialized_scanned_teacher(seed=16)
    readers = extract_frozen_memory_readers(teacher, (0,))
    compressor = LocalCompressor(
        dim=1024, hidden=2, rngs=nnx.Rngs(17), dtype=jnp.float32
    )
    encoded = jax.random.normal(jax.random.key(18), (1, 32, 16, 1024))
    query = jax.random.normal(jax.random.key(19), (1, 1, 1, 1024))
    target = jax.random.normal(jax.random.key(20), (1, 1, 1, 1024))
    valid = jnp.ones((1, 1), dtype=jnp.bool_)

    target_grad = jax.grad(
        lambda teacher_reads: read_distill_loss(target + 1, teacher_reads, valid)
    )(target)
    grads = nnx.grad(
        lambda module: compressor_distill_loss(
            module, encoded, query, target, valid, readers
        )
    )(compressor)
    gradient_paths = set(grads.flat_state())

    np.testing.assert_array_equal(target_grad, jnp.zeros_like(target_grad))
    assert gradient_paths == {
        ("fc1", "bias"),
        ("fc1", "kernel"),
        ("fc2", "bias"),
        ("fc2", "kernel"),
    }
    assert jnp.linalg.norm(grads.fc2.kernel.value) > 0


def test_stage_a_state_overfits_one_stopped_capture():
    teacher = _initialized_scanned_teacher(seed=21)
    readers = extract_frozen_memory_readers(teacher, (0,))
    compressor = LocalCompressor(
        dim=1024, hidden=2, rngs=nnx.Rngs(22), dtype=jnp.float32
    )
    state = init_compressor_train_state(
        compressor,
        optax.adamw(learning_rate=3e-3, weight_decay=0.0),
    )
    captured = {
        "encoded_frames": jax.random.normal(
            jax.random.key(23), (1, 32, 16, 1024)
        ),
        "query_inputs": jax.random.normal(
            jax.random.key(24), (1, 1, 1, 1024)
        ),
        "teacher_reads": jax.random.normal(
            jax.random.key(25), (1, 1, 1, 1024)
        ),
        "query_valid": jnp.ones((1, 1), dtype=jnp.bool_),
    }
    reader_before = jax.tree.map(np.asarray, readers)
    update = jax.jit(compressor_train_step)

    losses = []
    for _ in range(4):
        state, metrics = update(state, captured, readers)
        losses.append(float(metrics["loss"]))

    assert set(state.params.flat_state()) == {
        ("fc1", "bias"),
        ("fc1", "kernel"),
        ("fc2", "bias"),
        ("fc2", "kernel"),
    }
    assert losses[-1] < losses[0]
    jax.tree.map(
        np.testing.assert_array_equal,
        readers,
        reader_before,
    )
