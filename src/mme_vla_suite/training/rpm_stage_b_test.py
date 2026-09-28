import ast
import dataclasses
import inspect
import textwrap

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import tyro

from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0
from mme_vla_suite.training.config import RPMFullCheckpointWeightLoader
from mme_vla_suite.training.config import get_config


def _path_strings(state):
    return {"/".join(map(str, path)) for path in state.flat_state()}


def test_stage_b_config_trains_only_rpm_and_action_modules():
    config = get_config("rpm_stage_b")
    assert isinstance(config.weight_loader, RPMFullCheckpointWeightLoader)
    assert config.weight_loader.params_path is tyro.MISSING
    assert config.model.history_config == "perceptual-rpm-modul.yaml"

    model = nnx.eval_shape(config.model.create, jax.random.key(0))
    params = nnx.state(model)
    frozen = _path_strings(params.filter(config.freeze_filter))
    trainable = _path_strings(params.filter(config.trainable_filter))

    assert "PaliGemma/img/embedding/kernel" in frozen
    assert "PaliGemma/llm/embedder/input_embedding" in frozen
    assert "PaliGemma/llm/layers/mlp/linear" in frozen
    assert "mem_encoder/compressor/fc2/kernel" in trainable
    assert "mem_encoder/feature_encoder/encoder_static/kernel" in trainable
    assert "PaliGemma/llm/layers/mem_attn/q_einsum_mem/w" in trainable
    assert "PaliGemma/llm/layers/mem_rms_norm_ffn/Dense_0/kernel" in trainable
    assert "PaliGemma/llm/layers/mlp_1/linear" in trainable
    assert "action_out_proj/kernel" in trainable
    assert not any(path.startswith("PaliGemma/img/") for path in trainable)
    assert "PaliGemma/llm/layers/mlp/linear" not in trainable


def test_stage_b_uses_action_loss_without_distillation():
    source = inspect.getsource(HistoryPi0.compute_loss)

    assert "v_t - u_t" in source
    assert "read_distill" not in source
    assert "capture_teacher_reads" not in source


def test_stage_b_action_loss_and_filtered_gradient_abstract_smoke():
    config = get_config("rpm_stage_b")
    model_config = dataclasses.replace(
        config.model,
        history_config=get_history_config(config.model.history_config),
    )
    observation, actions = model_config.inputs_spec(batch_size=1)

    def one_step(rng, obs, action):
        model = model_config.create(rng)

        def objective(candidate):
            loss, _ = candidate.compute_loss(rng, obs, action, train=True)
            return jnp.mean(loss)

        return nnx.value_and_grad(
            objective,
            argnums=nnx.DiffState(0, config.trainable_filter),
        )(model)

    loss, grads = jax.eval_shape(
        one_step,
        jax.random.key(1),
        observation,
        actions,
    )

    assert loss.shape == ()
    assert len(grads.flat_state()) == 33
    assert ("mem_encoder", "compressor", "fc2", "kernel") in grads.flat_state()
    assert ("PaliGemma", "llm", "layers", "mlp", "linear") not in grads.flat_state()


def test_sample_actions_does_not_encode_memory_inside_denoise_loop():
    source = textwrap.dedent(inspect.getsource(HistoryPi0.sample_actions))
    function = ast.parse(source).body[0]
    step = next(
        node
        for node in function.body
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    )
    calls_inside_step = [
        node
        for node in ast.walk(step)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "embed_memory"
    ]

    assert calls_inside_step == []
