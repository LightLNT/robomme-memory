import re

import flax.traverse_util

import openpi.shared.array_typing as at

_RPM_NEW_PARAM = re.compile(
    r"(?:.*/)?mem_encoder/compressor/(?:fc1|fc2)/(?:kernel|bias)"
)


def merge_rpm_checkpoint_params(
    loaded_params: at.Params,
    reference_params: at.Params,
) -> at.Params:
    """Merge a baseline checkpoint, allowing only RPM compressor weights to be new."""
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    flat_reference = flax.traverse_util.flatten_dict(reference_params, sep="/")

    unexpected = sorted(set(flat_loaded) - set(flat_reference))
    if unexpected:
        raise ValueError(f"Checkpoint contains unexpected parameters: {unexpected[:10]}")

    missing = sorted(set(flat_reference) - set(flat_loaded))
    disallowed_missing = [key for key in missing if not _RPM_NEW_PARAM.fullmatch(key)]
    if disallowed_missing:
        raise ValueError(
            f"Checkpoint is missing existing model parameters: {disallowed_missing[:10]}"
        )

    shape_mismatches = sorted(
        key
        for key in set(flat_loaded) & set(flat_reference)
        if flat_loaded[key].shape != flat_reference[key].shape
    )
    if shape_mismatches:
        details = [
            (
                key,
                flat_loaded[key].shape,
                flat_reference[key].shape,
            )
            for key in shape_mismatches[:10]
        ]
        raise ValueError(f"Checkpoint parameter shapes do not match: {details}")

    merged = {}
    for key, reference_value in flat_reference.items():
        if key in flat_loaded:
            loaded_value = flat_loaded[key]
            merged[key] = (
                loaded_value.astype(reference_value.dtype)
                if loaded_value.dtype != reference_value.dtype
                else loaded_value
            )
        else:
            merged[key] = reference_value

    return flax.traverse_util.unflatten_dict(merged, sep="/")


def load_exact_checkpoint_params(
    loaded_params: at.Params,
    reference_params: at.Params,
) -> at.Params:
    """Validate and cast a complete checkpoint without filling any parameters."""
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    flat_reference = flax.traverse_util.flatten_dict(reference_params, sep="/")
    missing = sorted(set(flat_reference) - set(flat_loaded))
    unexpected = sorted(set(flat_loaded) - set(flat_reference))
    if missing or unexpected:
        raise ValueError(
            "Complete checkpoint parameter keys do not match: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    mismatches = sorted(
        key
        for key in flat_reference
        if flat_loaded[key].shape != flat_reference[key].shape
    )
    if mismatches:
        raise ValueError(f"Complete checkpoint shape mismatch: {mismatches[:10]}")
    return flax.traverse_util.unflatten_dict(
        {
            key: (
                flat_loaded[key].astype(reference.dtype)
                if flat_loaded[key].dtype != reference.dtype
                else flat_loaded[key]
            )
            for key, reference in flat_reference.items()
        },
        sep="/",
    )


def merge_rpm_stage_a_params(
    baseline_params: at.Params,
    compressor_params: at.Params,
    reference_params: at.Params,
) -> at.Params:
    """Merge a baseline and compressor-only Stage-A checkpoint into RPM params."""
    merged = merge_rpm_checkpoint_params(baseline_params, reference_params)
    flat_merged = flax.traverse_util.flatten_dict(merged, sep="/")
    flat_reference = flax.traverse_util.flatten_dict(reference_params, sep="/")
    flat_compressor = flax.traverse_util.flatten_dict(compressor_params, sep="/")

    expected_compressor_keys = {
        "fc1/kernel",
        "fc1/bias",
        "fc2/kernel",
        "fc2/bias",
    }
    actual_compressor_keys = set(flat_compressor)
    if actual_compressor_keys != expected_compressor_keys:
        missing = sorted(expected_compressor_keys - actual_compressor_keys)
        unexpected = sorted(actual_compressor_keys - expected_compressor_keys)
        raise ValueError(
            "Stage-A compressor checkpoint has invalid keys: "
            f"missing={missing}, unexpected={unexpected}"
        )

    target_paths = {}
    marker = "mem_encoder/compressor/"
    for path in flat_reference:
        if _RPM_NEW_PARAM.fullmatch(path):
            relative_path = path[path.index(marker) + len(marker) :]
            if relative_path in target_paths:
                raise ValueError(f"Duplicate RPM compressor target for {relative_path}")
            target_paths[relative_path] = path
    if set(target_paths) != expected_compressor_keys:
        raise ValueError(
            "RPM model reference does not contain the expected compressor parameters"
        )

    for relative_path, target_path in target_paths.items():
        source = flat_compressor[relative_path]
        reference = flat_reference[target_path]
        if source.shape != reference.shape:
            raise ValueError(
                f"Stage-A compressor shape mismatch for {relative_path}: "
                f"checkpoint={source.shape}, model={reference.shape}"
            )
        flat_merged[target_path] = (
            source.astype(reference.dtype)
            if source.dtype != reference.dtype
            else source
        )

    return flax.traverse_util.unflatten_dict(flat_merged, sep="/")
