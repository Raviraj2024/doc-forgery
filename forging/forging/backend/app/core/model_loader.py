from __future__ import annotations

import logging
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from app.core.config import Settings

try:
    import segmentation_models_pytorch as smp
except ImportError:  # pragma: no cover - exercised when deps are absent
    smp = None


class ModelLoader:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = self._resolve_device(settings.model_device)
        self.model: Any | None = None
        self.selected_encoder: str | None = None
        self.input_channels: int | None = None
        self.model_parameter_count: int | None = None
        self.model_loaded: bool = False
        self.load_error: str | None = None
        self.checkpoint_input_channels: int | None = None
        self.tried_architectures: list[str] = []
        self.checkpoint_sha256: str | None = None
        self.checkpoint_size_bytes: int | None = None

    def _resolve_device(self, preferred: str) -> str:
        if preferred == "cpu":
            return "cpu"
        if preferred == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _normalise_state_dict(self, raw: object) -> OrderedDict[str, torch.Tensor]:
        if isinstance(raw, OrderedDict):
            state_dict = raw
        elif isinstance(raw, dict):
            for key in ("state_dict", "model_state_dict", "model", "weights"):
                value = raw.get(key)
                if isinstance(value, (dict, OrderedDict)):
                    state_dict = OrderedDict(value)
                    break
            else:
                raise RuntimeError("Checkpoint dictionary does not contain a state dict.")
        else:
            raise RuntimeError(f"Unsupported checkpoint object type: {type(raw)!r}")

        cleaned = OrderedDict()
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith("module.") else key
            cleaned[new_key] = value
        return cleaned

    def _inspect_state_dict(self, state_dict: OrderedDict[str, torch.Tensor]) -> int | None:
        # MiT / SegFormer pattern (mit_b3, mit_b4, mit_b5)
        mit_key = "encoder.patch_embed1.proj.weight"
        if mit_key in state_dict and state_dict[mit_key].ndim == 4:
            return int(state_dict[mit_key].shape[1])
        # EfficientNet pattern (efficientnet-b2, b3, b4)
        eff_key = "encoder._conv_stem.weight"
        if eff_key in state_dict and state_dict[eff_key].ndim == 4:
            return int(state_dict[eff_key].shape[1])
        return None

    def _detect_encoder_family(self, state_dict: OrderedDict[str, torch.Tensor]) -> str:
        """Detect whether the checkpoint uses MiT or EfficientNet encoder."""
        if "encoder.patch_embed1.proj.weight" in state_dict:
            return "mit"
        if "encoder._conv_stem.weight" in state_dict:
            return "efficientnet"
        return "unknown"

    def _candidate_architectures(
        self,
        inferred_channels: int | None,
        encoder_family: str = "unknown",
    ) -> list[tuple[str, int]]:
        ch = inferred_channels or 13

        mit_candidates: list[tuple[str, int]] = [
            ("mit_b4", ch),
            ("mit_b5", ch),
            ("mit_b3", ch),
            ("mit_b2", ch),
        ]
        eff_candidates: list[tuple[str, int]] = [
            ("efficientnet-b3", ch),
            ("efficientnet-b4", ch),
            ("efficientnet-b2", ch),
        ]
        # Also try the PRD default of 12 channels for EfficientNet
        eff_12: list[tuple[str, int]] = [
            ("efficientnet-b4", 12),
            ("efficientnet-b3", 12),
        ]

        if encoder_family == "mit":
            ordered = mit_candidates + eff_candidates + eff_12
        elif encoder_family == "efficientnet":
            ordered = eff_candidates + eff_12 + mit_candidates
        else:
            ordered = mit_candidates + eff_candidates + eff_12

        merged: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for candidate in ordered:
            if candidate not in seen:
                merged.append(candidate)
                seen.add(candidate)
        return merged

    def _build_model(self, encoder_name: str, in_channels: int) -> Any:
        if smp is None:
            raise RuntimeError(
                "segmentation-models-pytorch is not installed. "
                "Install requirements before loading the checkpoint."
            )
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )

    def load(self) -> None:
        checkpoint_path = Path(self.settings.checkpoint_path)
        self.tried_architectures = []
        self.model_loaded = False
        self.load_error = None
        self.model = None
        self.selected_encoder = None
        self.input_channels = None
        self.model_parameter_count = None
        self.checkpoint_sha256 = None
        self.checkpoint_size_bytes = None

        if not checkpoint_path.exists():
            self.load_error = f"Checkpoint not found at {checkpoint_path}"
            self.logger.error(self.load_error)
            return

        self.checkpoint_sha256 = self._hash_checkpoint(checkpoint_path)
        self.checkpoint_size_bytes = checkpoint_path.stat().st_size

        try:
            raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = self._normalise_state_dict(raw_checkpoint)
        except Exception as exc:  # pragma: no cover - defensive path
            self.load_error = f"Failed to read checkpoint: {exc}"
            self.logger.exception(self.load_error)
            return

        self.checkpoint_input_channels = self._inspect_state_dict(state_dict)
        encoder_family = self._detect_encoder_family(state_dict)
        self.logger.info(
            "Checkpoint inspection complete",
            extra={
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_input_channels": self.checkpoint_input_channels,
                "encoder_family": encoder_family,
            },
        )

        errors: list[str] = []
        for encoder_name, in_channels in self._candidate_architectures(
            self.checkpoint_input_channels,
            encoder_family=encoder_family,
        ):
            candidate_name = f"{encoder_name}:{in_channels}"
            self.tried_architectures.append(candidate_name)
            try:
                model = self._build_model(encoder_name=encoder_name, in_channels=in_channels)
                model.load_state_dict(state_dict, strict=True)
                model = model.to(self.device)
                model.eval()

                self.model = model
                self.selected_encoder = encoder_name
                self.input_channels = in_channels
                self.model_parameter_count = sum(
                    parameter.numel() for parameter in model.parameters()
                )
                self.model_loaded = True
                self.logger.info(
                    "Segmentation checkpoint loaded",
                    extra={
                        "checkpoint_path": str(checkpoint_path),
                        "selected_encoder": encoder_name,
                        "input_channels": in_channels,
                        "device": self.device,
                    },
                )
                return
            except Exception as exc:
                errors.append(f"{candidate_name} -> {exc}")

        self.load_error = (
            "Failed to load checkpoint with supported architectures. "
            f"Tried: {', '.join(self.tried_architectures)}. "
            f"Errors: {' | '.join(errors)}"
        )
        self.logger.error(self.load_error)

    def _hash_checkpoint(self, checkpoint_path: Path) -> str:
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def predict(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.model_loaded or self.model is None:
            raise RuntimeError(self.load_error or "Segmentation model is not loaded.")
        with torch.no_grad():
            return self.model(tensor.to(self.device))

    def info(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.settings.checkpoint_path),
            "selected_encoder": self.selected_encoder,
            "input_channels": self.input_channels,
            "device": self.device,
            "model_parameter_count": self.model_parameter_count,
            "model_loaded": self.model_loaded,
            "load_error": self.load_error,
            "checkpoint_input_channels": self.checkpoint_input_channels,
            "tried_architectures": self.tried_architectures,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
        }
