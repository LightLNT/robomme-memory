import numpy as np
import pytest

from mme_vla_suite.training.rpm_weight_loader import load_exact_checkpoint_params
from mme_vla_suite.training.rpm_weight_loader import merge_rpm_checkpoint_params
from mme_vla_suite.training.rpm_weight_loader import merge_rpm_stage_a_params


def _reference_params():
    return {
        "PaliGemma": {"llm": {"kernel": np.zeros((2, 3), dtype=np.float32)}},
        "mem_encoder": {
            "feature_encoder": {
                "kernel": np.zeros((3, 4), dtype=np.float32),
            },
            "compressor": {
                "fc1": {
                    "kernel": np.ones((4, 2), dtype=np.float32),
                    "bias": np.ones((2,), dtype=np.float32),
                },
                "fc2": {
                    "kernel": np.zeros((2, 4), dtype=np.float32),
                    "bias": np.zeros((4,), dtype=np.float32),
                },
            },
        },
    }


def _baseline_checkpoint():
    return {
        "PaliGemma": {"llm": {"kernel": np.full((2, 3), 2, dtype=np.float32)}},
        "mem_encoder": {
            "feature_encoder": {
                "kernel": np.full((3, 4), 3, dtype=np.float32),
            },
        },
    }


def _compressor_checkpoint():
    return {
        "fc1": {
            "kernel": np.full((4, 2), 4, dtype=np.float32),
            "bias": np.full((2,), 5, dtype=np.float32),
        },
        "fc2": {
            "kernel": np.full((2, 4), 6, dtype=np.float32),
            "bias": np.full((4,), 7, dtype=np.float32),
        },
    }


def test_merge_allows_only_new_compressor_parameters():
    reference = _reference_params()
    merged = merge_rpm_checkpoint_params(_baseline_checkpoint(), reference)

    np.testing.assert_array_equal(
        merged["PaliGemma"]["llm"]["kernel"],
        _baseline_checkpoint()["PaliGemma"]["llm"]["kernel"],
    )
    np.testing.assert_array_equal(
        merged["mem_encoder"]["compressor"]["fc1"]["kernel"],
        reference["mem_encoder"]["compressor"]["fc1"]["kernel"],
    )


def test_merge_rejects_missing_existing_parameter():
    checkpoint = _baseline_checkpoint()
    del checkpoint["mem_encoder"]["feature_encoder"]["kernel"]

    with pytest.raises(ValueError, match="missing existing model parameters"):
        merge_rpm_checkpoint_params(checkpoint, _reference_params())


def test_merge_rejects_existing_parameter_shape_mismatch():
    checkpoint = _baseline_checkpoint()
    checkpoint["PaliGemma"]["llm"]["kernel"] = np.zeros((3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="shapes do not match"):
        merge_rpm_checkpoint_params(checkpoint, _reference_params())


def test_stage_b_merge_injects_only_stage_a_compressor():
    merged = merge_rpm_stage_a_params(
        _baseline_checkpoint(),
        _compressor_checkpoint(),
        _reference_params(),
    )

    np.testing.assert_array_equal(
        merged["PaliGemma"]["llm"]["kernel"],
        _baseline_checkpoint()["PaliGemma"]["llm"]["kernel"],
    )
    np.testing.assert_array_equal(
        merged["mem_encoder"]["compressor"]["fc2"]["kernel"],
        _compressor_checkpoint()["fc2"]["kernel"],
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda params: params["fc1"].pop("bias"),
        lambda params: params.update({"extra": np.zeros((1,), dtype=np.float32)}),
    ],
)
def test_stage_b_merge_rejects_invalid_compressor_keys(mutation):
    compressor = _compressor_checkpoint()
    mutation(compressor)

    with pytest.raises(ValueError, match="invalid keys"):
        merge_rpm_stage_a_params(
            _baseline_checkpoint(), compressor, _reference_params()
        )


def test_stage_b_merge_rejects_compressor_shape_mismatch():
    compressor = _compressor_checkpoint()
    compressor["fc2"]["kernel"] = np.zeros((4, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="compressor shape mismatch"):
        merge_rpm_stage_a_params(
            _baseline_checkpoint(), compressor, _reference_params()
        )


def test_exact_loader_rejects_missing_full_checkpoint_parameter():
    checkpoint = _reference_params()
    del checkpoint["mem_encoder"]["compressor"]["fc1"]["bias"]

    with pytest.raises(ValueError, match="keys do not match"):
        load_exact_checkpoint_params(checkpoint, _reference_params())


def test_exact_loader_casts_complete_checkpoint():
    checkpoint = _reference_params()
    checkpoint["PaliGemma"]["llm"]["kernel"] = checkpoint["PaliGemma"]["llm"][
        "kernel"
    ].astype(np.float64)

    loaded = load_exact_checkpoint_params(checkpoint, _reference_params())

    assert loaded["PaliGemma"]["llm"]["kernel"].dtype == np.float32
