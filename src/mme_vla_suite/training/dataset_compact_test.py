# ruff: noqa: SLF001

from types import SimpleNamespace

import numpy as np

from mme_vla_suite.dataset_builder.build_robomme_dataset import select_history_features
from mme_vla_suite.training.dataset import RoboMMEDataset


def test_select_history_features_preserves_full_payload():
    features = {
        "image_emb_8x8": np.zeros((1, 64, 2)),
        "image_emb_4x4": np.ones((1, 16, 2)),
        "image_emb_2x2": np.zeros((1, 4, 2)),
        "pos_emb_4x4": np.zeros((1, 16, 3)),
        "state_emb": np.arange(8),
    }

    assert select_history_features(features, "full") is features


def test_select_history_features_keeps_only_rpm_payload():
    features = {
        "image_emb_8x8": np.zeros((1, 64, 2)),
        "image_emb_4x4": np.ones((1, 16, 2)),
        "image_emb_2x2": np.zeros((1, 4, 2)),
        "pos_emb_4x4": np.zeros((1, 16, 3)),
        "state_emb": np.arange(8),
    }

    compact = select_history_features(features, "rpm_compact")

    assert set(compact) == {"image_emb_4x4", "state_emb"}
    np.testing.assert_array_equal(compact["image_emb_4x4"], features["image_emb_4x4"])
    np.testing.assert_array_equal(compact["state_emb"], features["state_emb"])


def test_compact_loader_injects_shared_position_embedding(tmp_path):
    feature_dir = tmp_path / "features"
    episode_dir = feature_dir / "episode_7"
    episode_dir.mkdir(parents=True)
    np.save(
        episode_dir / "token_emb_3.npy",
        {
            "image_emb_4x4": np.ones((1, 16, 2), dtype=np.float32),
            "state_emb": np.arange(8, dtype=np.float32),
        },
    )
    positions = np.arange(5 * 16 * 3, dtype=np.float32).reshape(5, 16, 3)
    np.save(feature_dir / "pos_emb_4x4.npy", positions)

    dataset = RoboMMEDataset.__new__(RoboMMEDataset)
    dataset.feature_dir = feature_dir
    dataset.feature_profile = "rpm_compact"
    dataset._compact_pos_emb_4x4 = None
    dataset.history_config = SimpleNamespace(representation_type="perceptual")

    result = dataset._gather_history_feat([3], epis_idx=7)

    assert isinstance(dataset._compact_pos_emb_4x4, np.memmap)
    np.testing.assert_array_equal(result[3]["pos_emb_4x4"], positions[3][None, ...])
