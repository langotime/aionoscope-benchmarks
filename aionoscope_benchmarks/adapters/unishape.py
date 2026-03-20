from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from .base import FrozenTimeSeriesAdapter


class _BaseUniShapeAdapter(FrozenTimeSeriesAdapter):
    source = "https://github.com/qianlima-lab/UniShape"
    import_path = "UniShape repo"
    env_name = "core"
    default_encode_batch_size = 512
    use_bfloat16_amp = True
    benchmark_sequence_length = 512
    benchmark_sequence_length_source = "official_unishape_scripts_resized_series_length"

    def __init__(self) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{self.model_name} requires CUDA for the official UniShape code path because "
                "TransformerEnc initializes cls_token with .cuda()"
            )

        repo_root = Path(__file__).resolve().parents[2]
        self.unishape_root = repo_root / "external" / "UniShape"
        if not self.unishape_root.is_dir():
            raise FileNotFoundError(f"Expected UniShape repo at {self.unishape_root}")
        sys.path.insert(0, str(self.unishape_root))

        self.model = self._build_model()
        self._load_checkpoint()
        self.model.eval()
        self.hidden_size = int(self._transformer_unit().cls_token.shape[0])
        self.num_hidden_layers = int(len(self._transformer_unit().transformer.layers))

    @property
    def available_layers(self) -> tuple[int, ...]:
        return tuple(range(self.num_hidden_layers + 1))

    def parameter_count_prefix_sources(self) -> dict[int, tuple[object, ...]]:
        sources: dict[int, tuple[object, ...]] = {0: self._layer_zero_parameter_sources()}
        for layer_index, (attn, ff) in enumerate(self._transformer_unit().transformer.layers, start=1):
            sources[int(layer_index)] = (attn, ff)
        return sources

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["hidden_size"] = int(self.hidden_size)
        payload["num_hidden_layers"] = int(self.num_hidden_layers)
        payload["representation_token"] = "cls_token"
        payload["checkpoint_origin"] = "official_repo_pretrained_model_ckpt"
        payload["preprocess"] = (
            "use the official UniShape multiscale tokenization path at the repository-published resized series length "
            "and read the CLS token after each transformer layer"
        )
        payload["layer_layout"] = (
            "layer 0 is the CLS token after positional encoding and before the first transformer block; "
            "layers 1..N are the CLS token after each transformer block"
        )
        return payload

    def _build_model(self):
        raise NotImplementedError

    def _checkpoint_path(self) -> Path:
        raise NotImplementedError

    def _transformer_unit(self):
        raise NotImplementedError

    def _build_hidden(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _layer_zero_parameter_sources(self) -> tuple[object, ...]:
        raise NotImplementedError

    def _load_checkpoint(self) -> None:
        checkpoint = torch.load(self._checkpoint_path(), map_location="cpu")
        state_dict = checkpoint["state_dict"]
        model_state = self.model.state_dict()
        filtered_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not key.startswith("backbone.") or key.startswith("backbone.fc."):
                continue
            model_key = key[len("backbone.") :]
            if model_key not in model_state:
                continue
            if tuple(value.shape) != tuple(model_state[model_key].shape):
                continue
            filtered_state[model_key] = value
        self.model.load_state_dict(filtered_state, strict=False)

    def _collect_transformer_layers(
        self,
        hidden_states: torch.Tensor,
        *,
        requested_layers: set[int],
    ) -> dict[int, torch.Tensor]:
        transformer_unit = self._transformer_unit()
        reps: dict[int, torch.Tensor] = {}
        if 0 in requested_layers:
            reps[0] = hidden_states[:, 0, :].float()
        for layer_index, (attn, ff) in enumerate(transformer_unit.transformer.layers, start=1):
            hidden_states = attn(hidden_states) + hidden_states
            hidden_states = ff(hidden_states) + hidden_states
            if layer_index in requested_layers:
                reps[int(layer_index)] = hidden_states[:, 0, :].float()
        return reps

    def forward_layer_dict(
        self,
        x: torch.Tensor,
        *,
        layers: tuple[int, ...] | None = None,
    ) -> dict[int, torch.Tensor]:
        requested_layers = set(int(layer) for layer in (layers or self.available_layers))
        self.validate_benchmark_input(x, channels=1)
        hidden_states = self._build_hidden(x.to(dtype=torch.float32))
        return self._collect_transformer_layers(hidden_states, requested_layers=requested_layers)


class UniShapeZeroShotAdapter(_BaseUniShapeAdapter):
    model_name = "UniShape-ZeroShot"
    model_slug = "UniShape-ZeroShot"
    checkpoint = "pretrained_model_ckpt/unishape_checkpoint_zeroshot.pth"

    def _build_model(self):
        from models.unishapemodel_zeroshot import UniShapeModel

        config = SimpleNamespace(window_size=16, stride=16)
        return UniShapeModel(
            config=config,
            series_size=512,
            in_channels=128,
            window_emb_dim=128,
            out_channels=10,
            window_size=16,
            stride=16,
            shape_alpha=0.01,
            shape_sparse_ratio=0.6,
            scale_len=4,
        )

    def _checkpoint_path(self) -> Path:
        return self.unishape_root / "pretrained_model_ckpt" / "unishape_checkpoint_zeroshot.pth"

    def _transformer_unit(self):
        return self.model.vit_unit

    def _layer_zero_parameter_sources(self) -> tuple[object, ...]:
        return (
            self.model.encoder_scale_list,
            self.model.inceptime_token,
            self.model.layer_norm_inc,
            self.model.attention_head,
            self._transformer_unit().pos_encoder,
        )

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["scale_len"] = 4
        payload["shape_sparse_ratio"] = 0.6
        return payload

    def _build_hidden(self, x: torch.Tensor) -> torch.Tensor:
        window_size_list = [64, 32, 16, 8]
        _, _, series_length = x.shape
        cls_tokens = None
        x_embed = None
        for scale_index in range(self.model.scale_len):
            if (series_length - 4) <= window_size_list[scale_index]:
                continue
            x_embed = self.model.encoder_scale_list[scale_index](x)
            if cls_tokens is None:
                input_x_embed = x_embed.permute(0, 2, 1)
            else:
                input_x_embed = torch.cat((cls_tokens, x_embed), dim=1).permute(0, 2, 1)
            cls_incep_token_list = self.model.inceptime_token(input_x_embed).permute(0, 2, 1)
            cls_incep_token_list = self.model.drop_token(
                self.model.layer_norm_inc(cls_incep_token_list)
            )
            cls_incep_token_list = self.model.act_gelu_inc(cls_incep_token_list)
            attn_x_score = self.model.attention_head(cls_incep_token_list)
            attn_shape_embeds = cls_incep_token_list * attn_x_score
            cls_tokens = torch.mean(attn_shape_embeds, dim=1).unsqueeze(1)
        if x_embed is None or cls_tokens is None:
            raise ValueError(f"{self.model_name} could not build multiscale tokens for input {tuple(x.shape)}")
        transformer_unit = self._transformer_unit()
        hidden_states = torch.cat([cls_tokens, x_embed], dim=1)
        return transformer_unit.pos_encoder(hidden_states.transpose(0, 1)).transpose(0, 1)


class UniShapeFineTuneAdapter(_BaseUniShapeAdapter):
    model_name = "UniShape-FineTune"
    model_slug = "UniShape-FineTune"
    checkpoint = "pretrained_model_ckpt/unishape_checkpoint_finetune.pth"

    def _build_model(self):
        from models.unishapemodel_finetune import UniShapeModel

        config = SimpleNamespace(window_size=16, stride=16)
        return UniShapeModel(
            config=config,
            series_size=512,
            in_channels=128,
            window_emb_dim=128,
            out_channels=10,
            window_size=16,
            stride=16,
            pre_training=False,
            shape_alpha=0.01,
            shape_ratio=0.6,
            scale_len=5,
        )

    def _checkpoint_path(self) -> Path:
        return self.unishape_root / "pretrained_model_ckpt" / "unishape_checkpoint_finetune.pth"

    def _transformer_unit(self):
        return self.model.transformer_enc

    def _layer_zero_parameter_sources(self) -> tuple[object, ...]:
        scale_index = int(self.model.scale_len) - 1
        scale_module = (
            self.model.unit_scale_list[scale_index]
            if scale_index == 4
            else self.model.unit_scale_list_finetune[scale_index]
        )
        return (
            scale_module,
            self.model.inceptime_token,
            self.model.layer_norm_inc,
            self.model.attention_head,
            self._transformer_unit().pos_encoder,
        )

    def adapter_metadata(self) -> dict[str, object]:
        payload = super().adapter_metadata()
        payload["scale_len"] = 5
        payload["shape_ratio"] = 0.6
        payload["checkpoint_type"] = "official_finetune_checkpoint"
        return payload

    def _build_hidden(self, x: torch.Tensor) -> torch.Tensor:
        scale_index = int(self.model.scale_len) - 1
        if scale_index == 4:
            x_embed = self.model.unit_scale_list[scale_index](x)
        else:
            x_embed = self.model.unit_scale_list_finetune[scale_index](x)
        cls_incep_token_list = self.model.inceptime_token(x_embed.permute(0, 2, 1)).permute(0, 2, 1)
        cls_incep_token_list = self.model.drop_token(self.model.layer_norm_inc(cls_incep_token_list))
        cls_incep_token_list = self.model.act_gelu_inc(cls_incep_token_list)
        attn_x_score = self.model.attention_head(cls_incep_token_list)
        attn_shape_embeds = cls_incep_token_list * attn_x_score
        cls_tokens = torch.mean(attn_shape_embeds, dim=1).unsqueeze(1)
        transformer_unit = self._transformer_unit()
        hidden_states = torch.cat([cls_tokens, x_embed], dim=1)
        return transformer_unit.pos_encoder(hidden_states.transpose(0, 1)).transpose(0, 1)
