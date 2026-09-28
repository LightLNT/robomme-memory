import numpy as np
import pytest

from mme_vla_suite.training.rpm_checkpoint import write_rpm_checkpoint
from openpi.models.model import restore_params


def test_write_rpm_checkpoint_uses_deployable_layout(tmp_path):
    params = {
        "model": {"kernel": np.arange(6, dtype=np.float32).reshape(2, 3)}
    }
    assets = tmp_path / "source_assets"
    assets.mkdir()
    assets.joinpath("marker.txt").write_text("assets")
    output = tmp_path / "rpm"

    step_dir = write_rpm_checkpoint(
        params,
        output,
        assets,
        "perceptual-rpm-modul.yaml",
        baseline_params_path="baseline/params",
        compressor_params_path="stage_a/params",
        overwrite=False,
    )

    restored = restore_params(step_dir / "params", restore_type=np.ndarray)
    np.testing.assert_array_equal(
        restored["model"]["kernel"], params["model"]["kernel"]
    )
    assert output.joinpath("history_config.txt").read_text() == (
        "perceptual-rpm-modul.yaml"
    )
    assert step_dir.joinpath("assets", "marker.txt").read_text() == "assets"
    assert "stage_a/params" in output.joinpath("rpm_merge_sources.txt").read_text()

    with pytest.raises(FileExistsError, match="already exists"):
        write_rpm_checkpoint(
            params,
            output,
            assets,
            "perceptual-rpm-modul.yaml",
            baseline_params_path="baseline/params",
            compressor_params_path="stage_a/params",
            overwrite=False,
        )
