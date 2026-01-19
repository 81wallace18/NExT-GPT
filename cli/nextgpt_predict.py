from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import torch

from nextgpt.model.multimodal_encoder.imagebind_processor import (
    ImageProcessor,
    VideoProcessor,
    AudioProcessor,
)
from nextgpt.runtime.generation import parse_generated_media
from nextgpt.runtime.io import collect_outputs, save_audios, save_images, save_videos
from nextgpt.runtime.loader import apply_overrides, load_config, load_model_bundle
from nextgpt.runtime.pipeline import build_generation_config, generate_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NExT-GPT inference")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--image", type=str, default=None, help="Image path or URL")
    parser.add_argument("--video", type=str, default=None, help="Video path")
    parser.add_argument("--audio", type=str, default=None, help="Audio path")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for media"
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Override model path"
    )
    parser.add_argument(
        "--model-base", type=str, default=None, help="Override model base"
    )
    parser.add_argument(
        "--model-name", type=str, default=None, help="Override model name"
    )
    parser.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    return parser


def main():
    args = build_parser().parse_args()
    config = load_config(args.config)

    overrides: Dict[str, Any] = {"paths": {}, "device": {}}
    if args.model_path:
        overrides["paths"]["model_path"] = args.model_path
    if args.model_base:
        overrides["paths"]["model_base"] = args.model_base
    if args.model_name:
        overrides["paths"]["model_name"] = args.model_name
    if args.load_4bit:
        overrides["device"]["load_4bit"] = True
    if args.load_8bit:
        overrides["device"]["load_8bit"] = True
    if args.device:
        overrides["device"]["device"] = args.device

    config = apply_overrides(config, overrides)

    (
        tokenizer,
        model,
        image_processor,
        video_processor,
        audio_processor,
        _,
        model_config,
    ) = load_model_bundle(config)

    if image_processor is None:
        image_processor = ImageProcessor()
    if video_processor is None:
        video_processor = VideoProcessor()
    if audio_processor is None:
        audio_processor = AudioProcessor()

    device = model.device

    images = None
    if args.image:
        image_data = image_processor.preprocess(args.image, return_tensors="pt")
        if image_data:
            images = image_data["pixel_values"].to(device=device, dtype=torch.float16)

    videos = None
    if args.video:
        video_data = video_processor.preprocess(args.video, return_tensors="pt")
        if video_data:
            videos = video_data["pixel_values"].to(device=device, dtype=torch.float16)

    audios = None
    if args.audio:
        audio_data = audio_processor.preprocess(args.audio, return_tensors="pt")
        if audio_data:
            audios = audio_data["pixel_values"].to(device=device, dtype=torch.float16)

    generation_cfg = build_generation_config(config.get("generation", {}))
    conv_mode = config.get("prompt", {}).get("conv_mode", "llava_v1")
    stop_token_ids = config.get("prompt", {}).get("stop_token_ids", [])

    _, outputs = generate_outputs(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        prompt=args.prompt,
        generation_cfg=generation_cfg,
        conv_mode=conv_mode,
        stop_token_ids=stop_token_ids,
        images=images,
        videos=videos,
        audios=audios,
    )

    media_outputs = parse_generated_media(outputs)

    output_dir = Path(
        args.output_dir or config.get("output", {}).get("assets_dir", "./assets")
    )
    if config.get("output", {}).get("save_images", True):
        save_images(media_outputs.get("images"), output_dir / "images", prefix="output")
    if config.get("output", {}).get("save_videos", True):
        save_videos(media_outputs.get("videos"), output_dir / "videos", prefix="output")
    if config.get("output", {}).get("save_audios", True):
        save_audios(media_outputs.get("audios"), output_dir / "audios", prefix="output")


if __name__ == "__main__":
    main()
