from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import FrozenTimeSeriesAdapter


class _BaseMoiraiAdapter(FrozenTimeSeriesAdapter):
    source = "https://github.com/SalesforceAIResearch/uni2ts"
    import_path = "uni2ts"
    env_name = "moirai"
    default_encode_batch_size = 64
    use_bfloat16_amp = True

    def __init__(self) -> None:
        super().__init__()
        self.model = self._load_model()
        self.model.eval()
        self.helper = self._build_helper()
        self.num_hidden_layers = int(self.model.num_layers)
        self.context_length = int(self._context_length())
        self.benchmark_sequence_length = int(self.context_length)
        self.benchmark_sequence_length_source = self._benchmark_sequence_length_source()

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_hidden_layers + 1))

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["context_length"] = int(self.context_length)
        payload["hidden_size"] = int(self._hidden_size())
        payload["num_hidden_layers"] = int(self.num_hidden_layers)
        payload["scaling"] = bool(self.model.scaling)
        payload["pooling"] = "mean over observed non-prediction tokens"
        payload["preprocess"] = (
            "expect exact model context length; transpose to [B,T,C]; pack tokens with the official forecast helper; "
            "mean-pool observed non-prediction tokens"
        )
        return payload

    def _load_model(self):
        raise NotImplementedError

    def _build_helper(self):
        raise NotImplementedError

    def _context_length(self) -> int:
        return int(self.model.max_seq_len)

    def _benchmark_sequence_length_source(self) -> str:
        return "model.max_seq_len"

    def _hidden_size(self) -> int:
        value = getattr(self.model, "d_model", None)
        if isinstance(value, int):
            return int(value)
        raise ValueError(f"{self.model_name} model is missing integer hidden size metadata")

    def _prepare_context(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.validate_benchmark_input(x, channels=1)
        past_target = x.transpose(1, 2).to(dtype=torch.float32)
        batch_size, _, channels = past_target.shape
        if channels != 1:
            raise ValueError(f"{self.model_name} adapter currently expects 1 channel, got {channels}")
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad = torch.zeros(
            batch_size,
            self.context_length,
            dtype=torch.bool,
            device=past_target.device,
        )
        return past_target, past_observed_target, past_is_pad

    def _pool_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        pool_mask: torch.Tensor,
    ) -> torch.Tensor:
        weight = pool_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        denom = weight.sum(dim=1).clamp_min(1.0)
        return ((hidden_states * weight).sum(dim=1) / denom).float()

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        raise NotImplementedError


class _Moirai1BaseAdapter(_BaseMoiraiAdapter):
    patch_size: int

    def _load_model(self):
        from uni2ts.model.moirai import MoiraiModule

        model = MoiraiModule.from_pretrained(self.checkpoint)
        self.patch_size = int(min(model.patch_sizes))
        return model

    def _build_helper(self):
        from uni2ts.model.moirai.forecast import MoiraiForecast

        return MoiraiForecast(
            prediction_length=1,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=self._context_length(),
            module=self.model,
            patch_size=self.patch_size,
            num_samples=1,
        )

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["patch_size"] = int(self.patch_size)
        payload["patch_sizes"] = [int(size) for size in self.model.patch_sizes]
        payload["layer_layout"] = (
            "layer 0 is the encoder input embedding stream; "
            "layers 1..N-1 are transformer block outputs; "
            "layer N is the final encoder block output"
        )
        return payload

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        from uni2ts.common.torch_util import mask_fill, packed_attention_mask

        requested_layers = set(int(layer) for layer in (layers or self.available_layers))
        past_target, past_observed_target, past_is_pad = self._prepare_context(x)
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self.helper._convert(
            self.patch_size,
            past_target,
            past_observed_target,
            past_is_pad,
        )
        patch_size = torch.full_like(time_id, fill_value=self.patch_size)
        loc, scale = self.model.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale
        hidden_states = self.model.in_proj(scaled_target, patch_size)
        hidden_states = mask_fill(hidden_states, prediction_mask, self.model.mask_encoding.weight)
        attention_mask = packed_attention_mask(sample_id)
        pool_mask = observed_mask.any(dim=-1) & ~prediction_mask

        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            reps[0] = self._pool_hidden(hidden_states, pool_mask=pool_mask)
        for layer_index, layer in enumerate(self.model.encoder.layers, start=1):
            hidden_states = layer(
                hidden_states,
                attention_mask,
                var_id=variate_id,
                time_id=time_id,
            )
            if layer_index in requested_layers:
                reps[int(layer_index)] = self._pool_hidden(hidden_states, pool_mask=pool_mask)
        return reps


class _Moirai2BaseAdapter(_BaseMoiraiAdapter):
    patch_size: int

    def _load_model(self):
        from uni2ts.model.moirai2 import Moirai2Module

        model = Moirai2Module.from_pretrained(self.checkpoint)
        self.patch_size = int(model.patch_size)
        return model

    def _build_helper(self):
        from uni2ts.model.moirai2 import Moirai2Forecast

        return Moirai2Forecast(
            prediction_length=1,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=self._context_length(),
            module=self.model,
        )

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["patch_size"] = int(self.patch_size)
        payload["num_predict_token"] = int(self.model.num_predict_token)
        payload["quantile_levels"] = [float(level) for level in self.model.quantile_levels]
        payload["layer_layout"] = (
            "layer 0 is the encoder input embedding stream; "
            "layers 1..N-1 are transformer block outputs; "
            "layer N is the final encoder block output"
        )
        return payload

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        from uni2ts.common.torch_util import packed_causal_attention_mask

        requested_layers = set(int(layer) for layer in (layers or self.available_layers))
        past_target, past_observed_target, past_is_pad = self._prepare_context(x)
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self.helper._convert(
            self.patch_size,
            past_target,
            past_observed_target,
            past_is_pad,
        )
        loc, scale = self.model.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale
        input_tokens = torch.cat([scaled_target, observed_mask.to(torch.float32)], dim=-1)
        hidden_states = self.model.in_proj(input_tokens)
        attention_mask = packed_causal_attention_mask(sample_id, time_id)
        pool_mask = observed_mask.any(dim=-1) & ~prediction_mask

        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            reps[0] = self._pool_hidden(hidden_states, pool_mask=pool_mask)
        for layer_index, layer in enumerate(self.model.encoder.layers, start=1):
            hidden_states = layer(
                hidden_states,
                attention_mask,
                var_id=variate_id,
                time_id=time_id,
            )
            if layer_index in requested_layers:
                reps[int(layer_index)] = self._pool_hidden(hidden_states, pool_mask=pool_mask)
        return reps


class _MoiraiMoEBaseAdapter(_BaseMoiraiAdapter):
    patch_size: int

    def _load_model(self):
        from uni2ts.model.moirai_moe import MoiraiMoEModule

        model = MoiraiMoEModule.from_pretrained(self.checkpoint)
        self.patch_size = int(min(model.patch_sizes))
        return model

    def _build_helper(self):
        from uni2ts.model.moirai_moe import MoiraiMoEForecast

        return MoiraiMoEForecast(
            prediction_length=1,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=self._context_length(),
            module=self.model,
            patch_size=self.patch_size,
            num_samples=1,
        )

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["patch_size"] = int(self.patch_size)
        payload["patch_sizes"] = [int(size) for size in self.model.patch_sizes]
        payload["layer_layout"] = (
            "layer 0 is the encoder input embedding stream after the official MoE projections; "
            "layers 1..N-1 are transformer block outputs; "
            "layer N is the final encoder block output"
        )
        return payload

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        from uni2ts.common.torch_util import packed_causal_attention_mask

        requested_layers = set(int(layer) for layer in (layers or self.available_layers))
        past_target, past_observed_target, past_is_pad = self._prepare_context(x)
        (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self.helper._convert(
            self.patch_size,
            past_target,
            past_observed_target,
            past_is_pad,
        )
        patch_size = torch.full_like(time_id, fill_value=self.patch_size)
        loc, scale = self.model.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale
        in_reprs = self.model.in_proj(scaled_target, patch_size)
        in_reprs = F.silu(in_reprs)
        in_reprs = self.model.feat_proj(in_reprs, patch_size)
        res_reprs = self.model.res_proj(scaled_target, patch_size)
        hidden_states = in_reprs + res_reprs
        attention_mask = packed_causal_attention_mask(sample_id, time_id)
        pool_mask = observed_mask.any(dim=-1) & ~prediction_mask

        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            reps[0] = self._pool_hidden(hidden_states, pool_mask=pool_mask)
        for layer_index, layer in enumerate(self.model.encoder.layers, start=1):
            hidden_states = layer(
                hidden_states,
                attention_mask,
                var_id=variate_id,
                time_id=time_id,
                centroid=self.model.encoder.centroid[layer_index - 1],
            )
            if layer_index in requested_layers:
                reps[int(layer_index)] = self._pool_hidden(hidden_states, pool_mask=pool_mask)
        return reps


class Moirai10RSmallAdapter(_Moirai1BaseAdapter):
    model_name = "Moirai-1.0-R-Small"
    model_slug = "Moirai-1.0-R-Small"
    checkpoint = "Salesforce/moirai-1.0-R-small"


class Moirai10RBaseAdapter(_Moirai1BaseAdapter):
    model_name = "Moirai-1.0-R-Base"
    model_slug = "Moirai-1.0-R-Base"
    checkpoint = "Salesforce/moirai-1.0-R-base"


class Moirai10RLargeAdapter(_Moirai1BaseAdapter):
    model_name = "Moirai-1.0-R-Large"
    model_slug = "Moirai-1.0-R-Large"
    checkpoint = "Salesforce/moirai-1.0-R-large"
    default_encode_batch_size = 32


class Moirai11RSmallAdapter(_Moirai1BaseAdapter):
    model_name = "Moirai-1.1-R-Small"
    model_slug = "Moirai-1.1-R-Small"
    checkpoint = "Salesforce/moirai-1.1-R-small"


class Moirai11RBaseAdapter(_Moirai1BaseAdapter):
    model_name = "Moirai-1.1-R-Base"
    model_slug = "Moirai-1.1-R-Base"
    checkpoint = "Salesforce/moirai-1.1-R-base"


class Moirai11RLargeAdapter(_Moirai1BaseAdapter):
    model_name = "Moirai-1.1-R-Large"
    model_slug = "Moirai-1.1-R-Large"
    checkpoint = "Salesforce/moirai-1.1-R-large"
    default_encode_batch_size = 32


class Moirai20RSmallAdapter(_Moirai2BaseAdapter):
    model_name = "Moirai-2.0-R-Small"
    model_slug = "Moirai-2.0-R-Small"
    checkpoint = "Salesforce/moirai-2.0-R-small"
    default_encode_batch_size = 32


class MoiraiMoE10RSmallAdapter(_MoiraiMoEBaseAdapter):
    model_name = "Moirai-MoE-1.0-R-Small"
    model_slug = "Moirai-MoE-1.0-R-Small"
    checkpoint = "Salesforce/moirai-moe-1.0-R-small"


class MoiraiMoE10RBaseAdapter(_MoiraiMoEBaseAdapter):
    model_name = "Moirai-MoE-1.0-R-Base"
    model_slug = "Moirai-MoE-1.0-R-Base"
    checkpoint = "Salesforce/moirai-moe-1.0-R-base"
    default_encode_batch_size = 32
