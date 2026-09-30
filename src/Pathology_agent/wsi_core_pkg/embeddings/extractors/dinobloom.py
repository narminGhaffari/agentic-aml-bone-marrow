"""
DinoBloom extractors for WSI embeddings.

Ported from the official DinoBloom Hugging Face usage example:
https://huggingface.co/MarrLab/DinoBloom
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
import tempfile
from tempfile import NamedTemporaryFile
from typing import Any
import urllib.request

from .. import Extractor

__license__ = "Apache-2.0"

_HF_REPO_ID = "MarrLab/DinoBloom"
_INPUT_SIZE = 224
_PATCH_SIZE = 14
_NUM_TOKENS = 1 + (_INPUT_SIZE // _PATCH_SIZE) ** 2
_STAMP_DINOBLOOM_URL = "https://zenodo.org/records/10908163/files/DinoBloom-S.pth"
_STAMP_DINOBLOOM_SHA256 = "c2f7990b003e89bcece80e379fb8fe0ba2ec392ce19b286e8a294abd99568e44"

_MODEL_VARIANTS: dict[str, dict[str, Any]] = {
    "s": {
        "identifier": "DinoBloom-S",
        "dinov2_model": "dinov2_vits14",
        "checkpoint": "pytorch_model_s.bin",
        "embed_dim": 384,
    },
    "b": {
        "identifier": "DinoBloom-B",
        "dinov2_model": "dinov2_vitb14",
        "checkpoint": "pytorch_model_b.bin",
        "embed_dim": 768,
    },
    "l": {
        "identifier": "DinoBloom-L",
        "dinov2_model": "dinov2_vitl14",
        "checkpoint": "pytorch_model_l.bin",
        "embed_dim": 1024,
    },
    "g": {
        "identifier": "DinoBloom-G",
        "dinov2_model": "dinov2_vitg14",
        "checkpoint": "pytorch_model_g.bin",
        "embed_dim": 1536,
    },
}


def _stamp_cache_dir() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "stamp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _file_digest(file: str | Path) -> str:
    with open(file, "rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def _model_cache_root() -> Path:
    raw = os.getenv("SLIDE_AGENT_MODEL_CACHE_DIR", "").strip()
    if raw:
        root = Path(raw).expanduser()
    else:
        root = Path(tempfile.gettempdir()) / "slide-agent-model-cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _normalize_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and "teacher" in checkpoint and isinstance(checkpoint["teacher"], dict):
        state_dict: dict[str, Any] = {}
        for key, value in checkpoint["teacher"].items():
            if "dino_head" in key or "ibot_head" in key:
                continue
            state_dict[key.replace("backbone.", "")] = value
        return state_dict

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]

    if isinstance(checkpoint, dict) and checkpoint:
        return checkpoint

    raise TypeError("Unsupported DinoBloom checkpoint format.")


@lru_cache(maxsize=1)
def _load_stamp_dinobloom_assets() -> tuple[Any, Any]:
    """Load DinoBloom-S exactly like STAMP's `dino_bloom()` extractor."""
    try:
        import torch
        from torch import nn
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DinoBloom dependencies not installed. "
            "Please install with `pip install torch torchvision`."
        ) from exc

    model_file = _stamp_cache_dir() / "dinobloom-s.pth"
    if not model_file.exists():
        with NamedTemporaryFile(dir=model_file.parent, delete=False) as tmp_model_file:
            urllib.request.urlretrieve(_STAMP_DINOBLOOM_URL, tmp_model_file.name)
            assert _file_digest(tmp_model_file.name) == _STAMP_DINOBLOOM_SHA256, (
                "unexpected DinoBloom-S model weights"
            )
            Path(tmp_model_file.name).rename(model_file)

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    pretrained = torch.load(model_file, map_location=torch.device("cpu"))
    state_dict = {}
    for key, value in pretrained["teacher"].items():
        if "dino_head" in key or "ibot_head" in key:
            continue
        state_dict[key.replace("backbone.", "")] = value

    # STAMP uses the DinoBloom-S ViT-S/14 position embedding for 224x224 tiles.
    model.pos_embed = nn.Parameter(torch.zeros(1, 257, _MODEL_VARIANTS["s"]["embed_dim"]))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return model, transform


@lru_cache(maxsize=4)
def _load_dinobloom_assets(variant: str) -> tuple[Any, Any]:
    if variant == "s":
        return _load_stamp_dinobloom_assets()

    try:
        import torch
        from huggingface_hub import hf_hub_download
        from torch import nn
        from torchvision import transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DinoBloom dependencies not installed. "
            "Please install with `pip install torch torchvision huggingface_hub`."
        ) from exc

    if variant not in _MODEL_VARIANTS:
        raise ValueError(f"Unknown DinoBloom variant: {variant!r}")

    config = _MODEL_VARIANTS[variant]
    cache_root = _model_cache_root()
    torch_hub_dir = cache_root / "torch_hub"
    hf_cache_dir = cache_root / "huggingface"
    torch_hub_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    torch.hub.set_dir(str(torch_hub_dir))
    model = torch.hub.load("facebookresearch/dinov2", config["dinov2_model"])
    checkpoint_path = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=config["checkpoint"],
        cache_dir=str(hf_cache_dir),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.pos_embed = nn.Parameter(torch.zeros(1, _NUM_TOKENS, config["embed_dim"]))
    model.load_state_dict(_normalize_state_dict(checkpoint), strict=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, transform


def _build_dinobloom_extractor(variant: str, identifier: str) -> Extractor[Any]:
    model, transform = _load_dinobloom_assets(variant)
    return Extractor(
        model=model,
        transform=transform,
        identifier=identifier,
    )


def dinobloom(identifier: str = "dino-bloom") -> Extractor[Any]:
    return _build_dinobloom_extractor(variant="s", identifier=identifier)


def dinobloom_base(identifier: str = "DinoBloom-B") -> Extractor[Any]:
    return _build_dinobloom_extractor(variant="b", identifier=identifier)


def dinobloom_large(identifier: str = "DinoBloom-L") -> Extractor[Any]:
    return _build_dinobloom_extractor(variant="l", identifier=identifier)


def dinobloom_giant(identifier: str = "DinoBloom-G") -> Extractor[Any]:
    return _build_dinobloom_extractor(variant="g", identifier=identifier)


def dino_bloom(identifier: str = "dino-bloom") -> Extractor[Any]:
    return dinobloom(identifier=identifier)
