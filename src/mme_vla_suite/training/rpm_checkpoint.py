"""Materialization helpers for complete RPM checkpoints."""

import pathlib
import shutil

import numpy as np
import orbax.checkpoint as ocp

from openpi.models.model import restore_params
import openpi.shared.array_typing as at

_REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _prepare_output_dir(output_dir: pathlib.Path, *, overwrite: bool):
    output_dir = output_dir.resolve()
    unsafe_targets = {
        pathlib.Path("/").resolve(),
        pathlib.Path.home().resolve(),
        pathlib.Path.cwd().resolve(),
        _REPOSITORY_ROOT,
    }
    if output_dir in unsafe_targets:
        raise ValueError(f"Refusing to use broad output directory: {output_dir}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}; pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    return output_dir


def write_rpm_checkpoint(
    params,
    output_dir: pathlib.Path,
    assets_dir: pathlib.Path,
    history_config: str,
    *,
    baseline_params_path: str,
    compressor_params_path: str,
    overwrite: bool,
):
    """Write and verify the repository's deployable checkpoint layout."""
    output_dir = _prepare_output_dir(output_dir, overwrite=overwrite)
    step_dir = output_dir / "0"
    params_dir = step_dir / "params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(params_dir, {"params": params})
    shutil.copytree(assets_dir, step_dir / "assets")
    output_dir.joinpath("history_config.txt").write_text(history_config)
    output_dir.joinpath("rpm_merge_sources.txt").write_text(
        f"baseline_params={baseline_params_path}\n"
        f"compressor_params={compressor_params_path}\n"
    )

    restored = restore_params(params_dir, restore_type=np.ndarray)
    at.check_pytree_equality(
        expected=params,
        got=restored,
        check_shapes=True,
        check_dtypes=True,
    )
    return step_dir
