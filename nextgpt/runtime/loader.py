from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from nextgpt.mm_utils import get_model_name_from_path
from nextgpt.model.builder import load_pretrained_model
from nextgpt.model.utils import auto_upgrade
from nextgpt.utils import disable_torch_init

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_overrides(
    config: Dict[str, Any], overrides: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if not overrides:
        return config

    def _deep_update(
        target: Dict[str, Any], incoming: Dict[str, Any]
    ) -> Dict[str, Any]:
        for key, value in incoming.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                _deep_update(target[key], value)
            else:
                target[key] = value
        return target

    return _deep_update(config, overrides)


def load_model_bundle(config: Dict[str, Any]):
    paths = config.get("paths", {})
    device_cfg = config.get("device", {})
    hf_cache = paths.get("hf_cache")
    if hf_cache:
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache

    disable_torch_init()

    model_path = paths.get("model_path")
    model_name = paths.get("model_name") or get_model_name_from_path(model_path or "")

    if model_path:
        auto_upgrade(model_path)

    optimization_cfg = config.get("optimization", {})

    return load_pretrained_model(
        model_path=model_path,
        model_base=paths.get("model_base"),
        model_name=model_name,
        load_8bit=device_cfg.get("load_8bit", False),
        load_4bit=device_cfg.get("load_4bit", False),
        device_map=device_cfg.get("device_map", "auto"),
        device=device_cfg.get("device", "cuda"),
        use_flash_attn=device_cfg.get("use_flash_attn", False),
        decoder_lazy_load=optimization_cfg.get("lazy_decoder_load", False),
        decoder_device=optimization_cfg.get("decoder_device"),
        decoder_dtype=optimization_cfg.get("decoder_dtype"),
        decoder_offload=optimization_cfg.get("decoder_offload"),
        decoder_attention_slicing=optimization_cfg.get("decoder_attention_slicing"),
        decoder_xformers=optimization_cfg.get("decoder_xformers"),
    )
