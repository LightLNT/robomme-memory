import flax.nnx as nnx
import jax
import jax.numpy as jnp

from mme_vla_suite.models.representation.mem_encoder import FeatureEncoder
from mme_vla_suite.shared.rpm_sampling import GROUPS
import openpi.shared.array_typing as at


class LocalCompressor(nnx.Module):
    def __init__(self, dim, hidden, rngs: nnx.Rngs, dtype):
        self.fc1 = nnx.Linear(4 * dim, hidden, rngs=rngs, dtype=dtype)
        self.fc2 = nnx.Linear(
            hidden,
            dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):
        cells = jnp.take(x, jnp.asarray(GROUPS), axis=-2)
        base = cells.mean(axis=-2)
        detail = cells - base[..., None, :]
        detail = detail.reshape(*base.shape[:-1], 4 * x.shape[-1])
        correction = self.fc2(jax.nn.gelu(self.fc1(detail)))
        return base + correction.astype(base.dtype)


def pack_encoded_frames(encoded_frames, compressor, mem_gather, mem_mask):
    """Pack fine and compressed frame tokens into the fixed read budget."""
    batch_size, max_frames, tokens_per_frame, width = encoded_frames.shape
    if tokens_per_frame != 16:
        raise ValueError("RPM expects 16 tokens per encoded frame")

    fine = encoded_frames.reshape(batch_size, max_frames * tokens_per_frame, width)
    coarse = compressor(encoded_frames).reshape(batch_size, max_frames * 4, width)
    candidates = jnp.concatenate([fine, coarse], axis=1)
    packed = jnp.take_along_axis(candidates, mem_gather[..., None], axis=1)
    return jnp.where(mem_mask[..., None], packed, 0)


class PerceptualMemory(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        self.config = config
        self.dtype = dtype

        self.mem_type = config.perceptual_memory.type
        self.is_multires = self.mem_type == "multires_frame_sampling"

        self.feature_encoder = FeatureEncoder(
            rngs=rngs,
            dtype=dtype,
            image_input_dim=self.config.memory_feature.img.input_dim,
            pos_input_dim=self.config.memory_feature.pos.input_dim,
            state_input_dim=self.config.memory_feature.state.input_dim,
            pos_output_dim=self.config.memory_feature.pos.hidden_dim,
            state_output_dim=self.config.memory_feature.state.hidden_dim,
            ouput_dim_for_recur=None,
            output_dim_for_percep=self.config.memory_token_dim,
            use_pos_emb=self.config.use_pos_emb,
            use_state_emb=self.config.use_state_emb,
        )
        if self.is_multires:
            self.compressor = LocalCompressor(
                dim=self.config.memory_token_dim,
                hidden=self.config.multires.compressor_hidden,
                rngs=rngs,
                dtype=dtype,
            )

    def __call__(
        self,
        static_image_emb: at.Float[at.Array, "b l d1"],
        static_pos_emb: at.Float[at.Array, "b l d2"],
        static_state_emb: at.Float[at.Array, "b l d3"],
        mem_gather: at.Int[at.Array, "b m"] | None = None,
        mem_mask: at.Bool[at.Array, "b m"] | None = None,
    ):
        # get memory tokens using feature encoder
        expected_length = (
            self.config.multires.raw_capacity
            if self.is_multires
            else self.config.budget
        )
        assert static_image_emb.shape[1] == expected_length

        hidden_states = self.feature_encoder.encode_perceptual_memory(
            static_image_emb, static_pos_emb, static_state_emb
        )

        if self.is_multires:
            if mem_gather is None or mem_mask is None:
                raise ValueError("RPM requires mem_gather and mem_mask")
            max_frames = self.config.multires.max_frames
            if expected_length != max_frames * self.config.token_per_image:
                raise ValueError("RPM raw capacity does not match its frame layout")

            encoded_frames = hidden_states.reshape(
                hidden_states.shape[0],
                max_frames,
                self.config.token_per_image,
                hidden_states.shape[-1],
            )
            hidden_states = pack_encoded_frames(
                encoded_frames, self.compressor, mem_gather, mem_mask
            )

        return hidden_states, None, None
