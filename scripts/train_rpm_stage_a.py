"""Train the RPM local compressor with frozen baseline memory reads.

Example:
  uv run scripts/train_rpm_stage_a.py rpm_stage_a --exp-name=pilot \
    --weight-loader.params-path=/path/to/perceptual-framesamp-modul/79999/params
"""

import dataclasses
import functools
import logging
import platform

import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import tqdm_loggable.auto as tqdm
import wandb

from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.representation.percep_mem import LocalCompressor
import mme_vla_suite.training.config as _config
import mme_vla_suite.training.dataloader as _data_loader
from mme_vla_suite.training.rpm_distillation import compressor_train_step
from mme_vla_suite.training.rpm_distillation import extract_frozen_memory_readers
from mme_vla_suite.training.rpm_distillation import init_compressor_train_state
import openpi.shared.array_typing as at
import openpi.training.checkpoints as _checkpoints
import openpi.training.optimizer as _optimizer


def _load_frozen_teacher(config: _config.RPMStageAConfig, rng):
    teacher = config.model.create(rng)
    reference = nnx.state(teacher)
    loaded = config.weight_loader.load(reference.to_pure_dict())
    at.check_pytree_equality(
        expected=reference.to_pure_dict(),
        got=loaded,
        check_shapes=True,
        check_dtypes=True,
    )
    reference.replace_by_pure_dict(loaded)
    nnx.update(teacher, reference)
    teacher.eval()
    return teacher


def _capture_batch(teacher, rng, observation, actions, layer_indices):
    noise_rng, time_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = (
        jax.random.beta(time_rng, 1.5, 1.0, actions.shape[:-2]) * 0.999
        + 0.001
    )
    return teacher.capture_teacher_reads(
        observation,
        actions,
        noise,
        time,
        layer_indices=layer_indices,
    )


def train_step(
    teacher_def,
    layer_indices,
    rng,
    teacher_params,
    state,
    batch,
):
    teacher = nnx.merge(teacher_def, teacher_params)
    observation, actions = batch
    captured = _capture_batch(teacher, rng, observation, actions, layer_indices)
    readers = extract_frozen_memory_readers(teacher.PaliGemma.llm, layer_indices)
    return compressor_train_step(state, captured, readers)


def _init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname).1s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main(config: _config.RPMStageAConfig):
    if not isinstance(config, _config.RPMStageAConfig):
        raise TypeError("train_rpm_stage_a.py requires the rpm_stage_a config")
    if jax.device_count() != 1:
        raise ValueError("RPM Stage A currently requires exactly one visible JAX device")

    _init_logging()
    logging.info("Running RPM Stage A on %s", platform.node())
    rng = jax.random.key(config.seed)
    teacher_rng, compressor_rng, train_rng = jax.random.split(rng, 3)

    checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=False,
    )
    wandb.init(
        mode="online" if config.wandb_enabled else "disabled",
        name=config.exp_name,
        project=config.project_name,
        config=dataclasses.asdict(config),
    )

    teacher_history = get_history_config(config.model.history_config)
    rpm_history = get_history_config(config.rpm_history_config)
    if (
        teacher_history.perceptual_memory.type != "frame_sampling"
        or teacher_history.integration_type != "modulation"
    ):
        raise ValueError("Stage-A teacher must use frame-sampling modulation")
    if rpm_history.perceptual_memory.type != "multires_frame_sampling":
        raise ValueError("Stage-A student config must use multires_frame_sampling")
    if tuple(rpm_history.multires.distill_layers) != config.distill_layers:
        raise ValueError("Stage-A layer config disagrees with the RPM model config")
    data_config = config.data.create(config.assets_dirs, config.model)
    data_loader = _data_loader.create_data_loader(
        config.dataset_path,
        data_config,
        history_config=config.model.history_config,
        sharding=None,
        shuffle=True,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    data_iter = iter(data_loader)

    teacher = _load_frozen_teacher(config, teacher_rng)
    teacher_def, teacher_params = nnx.split(teacher)
    compressor = LocalCompressor(
        dim=rpm_history.memory_token_dim,
        hidden=rpm_history.multires.compressor_hidden,
        rngs=nnx.Rngs(compressor_rng),
        dtype=jnp.float32,
    )
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule)
    state = init_compressor_train_state(compressor, tx)

    config.checkpoint_dir.joinpath("stage_a_config.txt").write_text(
        f"teacher_history_config={config.model.history_config}\n"
        f"rpm_history_config={config.rpm_history_config}\n"
        f"distill_layers={config.distill_layers}\n"
    )
    config.checkpoint_dir.joinpath("rpm_history_config.txt").write_text(
        config.rpm_history_config
    )
    logging.info(
        "Teacher frames=%d, distill layers=%s, compressor parameters=%d",
        teacher_history.budget // teacher_history.token_per_image,
        config.distill_layers,
        sum(value.size for value in jax.tree.leaves(state.params)),
    )

    compiled_step = jax.jit(
        functools.partial(train_step, teacher_def, config.distill_layers),
        donate_argnums=(2,),
    )
    infos = []
    progress = tqdm.tqdm(range(config.num_train_steps), dynamic_ncols=True)
    for step in progress:
        step_rng = jax.random.fold_in(train_rng, step)
        state, info = compiled_step(
            step_rng,
            teacher_params,
            state,
            next(data_iter),
        )
        infos.append(info)

        if step % config.log_interval == 0:
            stacked = common_utils.stack_forest(infos)
            reduced = jax.device_get(jax.tree.map(jnp.mean, stacked))
            progress.write(
                f"Step {step}: "
                + ", ".join(f"{name}={value:.6f}" for name, value in reduced.items())
            )
            wandb.log(reduced, step=step)
            infos = []

        if (
            step > 0 and step % config.save_interval == 0
        ) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, state, data_loader, step)

    checkpoint_manager.wait_until_finished()
    logging.info("RPM Stage A complete; compressor checkpoints saved to %s", config.checkpoint_dir)


if __name__ == "__main__":
    main(_config.cli())
