from __future__ import annotations

import torch

from .base import FrozenTimeSeriesAdapter


class TotoAdapter(FrozenTimeSeriesAdapter):
    model_name = "Toto-Open-Base-1.0"
    model_slug = "Toto-Open-Base-1.0"
    source = "https://github.com/DataDog/toto"
    checkpoint = "Datadog/Toto-Open-Base-1.0"
    import_path = "toto-ts"
    env_name = "toto"
    default_encode_batch_size = 512
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        from toto.model.toto import Toto

        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded = Toto.from_pretrained(self.checkpoint, map_location=device)
        self.model = loaded.model
        self.model.eval()
        self.benchmark_sequence_length = 4096
        self.benchmark_sequence_length_source = "official_toto_open_base_quickstart_context_length"

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(int(self.model.num_layers) + 1))

    def parameter_count_prefix_sources(self) -> dict[int, tuple[object, ...]]:
        return {
            0: (self.model.patch_embed,),
            **{
                int(layer_index): (layer,)
                for layer_index, layer in enumerate(self.model.transformer.layers, start=1)
            },
        }

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["patch_size"] = int(self.model.patch_embed.patch_size)
        payload["patch_stride"] = int(self.model.patch_embed.stride)
        payload["embed_dim"] = int(self.model.embed_dim)
        payload["preprocess"] = (
            "expect exact benchmark length 4096, matching the official Toto Open Base quick-start context; "
            "squeeze channel dimension; "
            "use Toto patch embedding and transformer; mean-pool valid patch tokens"
        )
        payload["layer_layout"] = (
            "layer 0 is the patch embedding stream; "
            "layers 1..N are transformer block outputs"
        )
        return payload

    def _prepare_inputs(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from toto.data.util.dataset import replace_extreme_values

        self.validate_benchmark_input(x, channels=1)
        series = replace_extreme_values(x.to(dtype=torch.float32))
        padding_mask = torch.ones_like(series, dtype=torch.bool)
        id_mask = torch.zeros_like(series, dtype=torch.long)
        return series, padding_mask, id_mask

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        from toto.model.attention import AttentionAxis

        requested_layers = set(layers or self.available_layers)
        series, padding_mask, id_mask = self._prepare_inputs(x)
        weights = torch.ones_like(series, dtype=series.dtype, device=series.device)
        scaled_inputs, _, _ = self.model.scaler(
            series,
            weights=weights,
            padding_mask=padding_mask,
            prefix_length=None,
        )
        hidden_states, reduced_id_mask = self.model.patch_embed(scaled_inputs, id_mask)

        patch_valid = padding_mask.unfold(
            dimension=-1,
            size=self.model.patch_embed.patch_size,
            step=self.model.patch_embed.stride,
        ).any(dim=-1)
        num_heads = int(self.model.transformer.layers[0].num_heads)
        spacewise_attention_mask = self.model.transformer._get_mask(
            num_heads=num_heads,
            dtype=hidden_states.dtype,
            id_mask=reduced_id_mask,
        )

        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            weight = patch_valid.unsqueeze(-1).to(dtype=hidden_states.dtype)
            denom = weight.sum(dim=(1, 2)).clamp_min(1.0)
            reps[0] = ((hidden_states * weight).sum(dim=(1, 2)) / denom).float()
        for block_index, layer in enumerate(self.model.transformer.layers):
            hidden_states = layer(
                block_index,
                hidden_states,
                None if layer.attention_axis == AttentionAxis.TIME else spacewise_attention_mask,
                None,
            )
            layer_index = block_index + 1
            if layer_index in requested_layers:
                weight = patch_valid.unsqueeze(-1).to(dtype=hidden_states.dtype)
                denom = weight.sum(dim=(1, 2)).clamp_min(1.0)
                reps[int(layer_index)] = ((hidden_states * weight).sum(dim=(1, 2)) / denom).float()
        return reps


class _Toto2Adapter(FrozenTimeSeriesAdapter):
    source = "https://github.com/DataDog/toto"
    import_path = (
        "toto-2 @ git+https://github.com/DataDog/toto.git#subdirectory=toto2; "
        "dd-unit-scaling @ git+https://github.com/DataDog/toto.git#subdirectory=dd_unit_scaling"
    )
    env_name = "toto"
    use_bfloat16_amp = True
    cpu_feature_cache_dtype = torch.float16
    benchmark_sequence_length = 512
    benchmark_sequence_length_source = "official_toto_2_0_quickstart_context_length"

    def __init__(self) -> None:
        super().__init__()
        from toto2 import Toto2Model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Toto2Model.from_pretrained(self.checkpoint, map_location=device)
        self.model.to(device)
        self.model.eval()

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(int(self.model.config.num_layers) + 1))

    def parameter_count_prefix_sources(self) -> dict[int, tuple[object, ...]]:
        total_layers = int(self.model.config.num_layers)
        sources: dict[int, tuple[object, ...]] = {0: (self.model.patch_proj,)}
        for layer_index, layer in enumerate(self.model.transformer.layers, start=1):
            layer_sources: tuple[object, ...] = (layer,)
            if layer_index == total_layers:
                layer_sources = layer_sources + (self.model.transformer.out_norm,)
            sources[int(layer_index)] = layer_sources
        return sources

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["patch_size"] = int(self.model.config.patch_size)
        payload["embed_dim"] = int(self.model.config.d_model)
        payload["preprocess"] = (
            "expect exact benchmark length 512, matching the official Toto 2.0 quick-start context; "
            "feed a single-variate target tensor with all observations marked present; "
            "use Toto 2.0 scaler, patch projection, and causal transformer; "
            "mean-pool valid patch tokens"
        )
        payload["layer_layout"] = (
            "layer 0 is the patch projection stream; "
            "layers 1..N-1 are transformer block outputs; "
            "layer N is the post-final-norm transformer output"
        )
        return payload

    def _prepare_inputs(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.validate_benchmark_input(x, channels=1)
        target = torch.nan_to_num(x.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        target_mask = torch.isfinite(x)
        cpm_mask = torch.ones_like(target_mask, dtype=torch.bool)
        series_ids = torch.zeros(target.shape[:-1], dtype=torch.long, device=target.device)
        return target, target_mask, cpm_mask, series_ids

    def _pool_patches(
        self,
        hidden_states: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        weight = (group_ids >= 0).unsqueeze(-1).to(dtype=hidden_states.dtype)
        denom = weight.sum(dim=(-3, -2)).clamp_min(1.0)
        return ((hidden_states * weight).sum(dim=(-3, -2)) / denom).float()

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        from einops import rearrange, reduce, repeat

        requested_layers = set(layers or self.available_layers)
        target, target_mask, cpm_mask, series_ids = self._prepare_inputs(x)
        observed_mask = target_mask & cpm_mask
        patch_size = int(self.model.config.patch_size)

        scaled_series, _, _ = self.model.scaler(target, observed_mask)
        scaled_series = scaled_series.asinh()
        hidden_states = self.model.patch_proj(
            torch.cat(
                [
                    rearrange(scaled_series, "... (seq patch) -> ... seq patch", patch=patch_size),
                    rearrange(
                        (~observed_mask).to(target.dtype),
                        "... (seq patch) -> ... seq patch",
                        patch=patch_size,
                    ),
                ],
                dim=-1,
            )
        )

        group_ids = repeat(series_ids, "... n_var -> ... n_var seq", seq=hidden_states.shape[-2]).clone()
        missing_patch = (
            reduce(target_mask, "... (seq patch) -> ... seq", "sum", patch=patch_size) == 0
        ) & (
            reduce(cpm_mask.to(dtype=torch.int64), "... (seq patch) -> ... seq", "prod", patch=patch_size)
            == 1
        )
        group_ids[missing_patch] = -1

        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            reps[0] = self._pool_patches(hidden_states, group_ids)

        transformer = self.model.transformer
        time_ids = torch.arange(hidden_states.shape[-2], device=hidden_states.device, dtype=torch.int32)
        time_layer_kwargs, var_layer_kwargs = transformer._sdpa_kwargs(
            hidden_states,
            time_ids,
            group_ids,
            has_missing_values=True,
        )

        num_series, seq_len = hidden_states.shape[-3], hidden_states.shape[-2]
        leading = hidden_states.shape[:-2]
        flat_states = rearrange(hidden_states, "... seq_len dim -> (...) seq_len dim")
        total_layers = int(self.model.config.num_layers)

        for block_index, layer in enumerate(transformer.layers):
            if transformer._if_variate_layer(block_index):
                flat_states = rearrange(flat_states, "(b n) s d -> (b s) n d", n=num_series)
                flat_states = layer(flat_states, **var_layer_kwargs)
                flat_states = rearrange(flat_states, "(b s) n d -> (b n) s d", s=seq_len)
            else:
                flat_states = layer(flat_states, seq_ids=time_ids, **time_layer_kwargs)

            layer_index = block_index + 1
            if layer_index in requested_layers:
                layer_states = flat_states.unflatten(0, leading)
                if layer_index == total_layers:
                    layer_states = transformer.out_norm(layer_states)
                reps[int(layer_index)] = self._pool_patches(layer_states, group_ids)

        return reps


class Toto2_4MAdapter(_Toto2Adapter):
    model_name = "Toto-2.0-4M"
    model_slug = "Toto-2.0-4M"
    checkpoint = "Datadog/Toto-2.0-4m"
    default_encode_batch_size = 512


class Toto2_22MAdapter(_Toto2Adapter):
    model_name = "Toto-2.0-22M"
    model_slug = "Toto-2.0-22M"
    checkpoint = "Datadog/Toto-2.0-22m"
    default_encode_batch_size = 256


class Toto2_313MAdapter(_Toto2Adapter):
    model_name = "Toto-2.0-313M"
    model_slug = "Toto-2.0-313M"
    checkpoint = "Datadog/Toto-2.0-313m"
    default_encode_batch_size = 128


class Toto2_1BAdapter(_Toto2Adapter):
    model_name = "Toto-2.0-1B"
    model_slug = "Toto-2.0-1B"
    checkpoint = "Datadog/Toto-2.0-1B"
    default_encode_batch_size = 128


class Toto2_25BAdapter(_Toto2Adapter):
    model_name = "Toto-2.0-2.5B"
    model_slug = "Toto-2.0-2.5B"
    checkpoint = "Datadog/Toto-2.0-2.5B"
    default_encode_batch_size = 128
