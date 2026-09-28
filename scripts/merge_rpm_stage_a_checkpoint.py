"""Materialize a complete RPM checkpoint from baseline and Stage-A weights.

Example:
  uv run scripts/merge_rpm_stage_a_checkpoint.py \
    --baseline-params-path=/path/to/perceptual-framesamp-modul/79999/params \
    --compressor-params-path=/path/to/rpm_stage_a/pilot/1999/params \
    --output-dir=runs/ckpts/rpm_stage_b/initialized
"""

import dataclasses
import pathlib

import flax.nnx as nnx
import jax
import tyro

from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
from mme_vla_suite.training.config import RPMStageBWeightLoader
from mme_vla_suite.training.rpm_checkpoint import write_rpm_checkpoint
import openpi.shared.array_typing as at


@dataclasses.dataclass(frozen=True)
class Args:
    baseline_params_path: str
    compressor_params_path: str
    output_dir: pathlib.Path
    assets_dir: pathlib.Path | None = None
    history_config: str = "perceptual-rpm-modul.yaml"
    overwrite: bool = False


def _resolve_assets_dir(args: Args) -> pathlib.Path:
    if args.assets_dir is not None:
        assets_dir = args.assets_dir.resolve()
    elif not args.baseline_params_path.startswith("gs://"):
        assets_dir = pathlib.Path(args.baseline_params_path).resolve().parent / "assets"
    else:
        raise ValueError("--assets-dir is required when the baseline checkpoint is remote")
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"Baseline assets directory not found: {assets_dir}")
    return assets_dir


def main(args: Args):
    assets_dir = _resolve_assets_dir(args)

    model_config = HistoryPi0Config(
        pi05=True,
        action_horizon=20,
        use_history=True,
        history_config=args.history_config,
        discrete_state_input=False,
    )
    shaped_model = nnx.eval_shape(model_config.create, jax.random.key(0))
    reference = nnx.state(shaped_model).to_pure_dict()
    loader = RPMStageBWeightLoader(
        baseline_params_path=args.baseline_params_path,
        compressor_params_path=args.compressor_params_path,
    )
    params = loader.load(reference)
    at.check_pytree_equality(
        expected=reference,
        got=params,
        check_shapes=True,
        check_dtypes=True,
    )

    step_dir = write_rpm_checkpoint(
        params,
        args.output_dir,
        assets_dir,
        args.history_config,
        baseline_params_path=args.baseline_params_path,
        compressor_params_path=args.compressor_params_path,
        overwrite=args.overwrite,
    )
    print(f"Complete RPM checkpoint written to {step_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
