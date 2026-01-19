from __future__ import annotations

import copy
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
import torch
from PIL import Image
from cog import BasePredictor

from nextgpt.model.multimodal_encoder.imagebind_processor import (
    AudioProcessor,
    ImageProcessor,
    VideoProcessor,
)
from nextgpt.runtime.generation import parse_generated_media
from nextgpt.runtime.io import save_audios, save_images, save_videos
from nextgpt.runtime.loader import apply_overrides, load_config, load_model_bundle
from nextgpt.runtime.pipeline import build_generation_config, generate_outputs


def load_image(image_file: str) -> Image.Image:
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")


class Predictor(BasePredictor):
    def setup(
        self, model_base, model_name, model_path, load_8bit=False, load_4bit=False
    ) -> None:
        config = load_config()
        overrides = {
            "paths": {
                "model_base": model_base,
                "model_name": model_name,
                "model_path": model_path,
            },
            "device": {
                "load_8bit": load_8bit,
                "load_4bit": load_4bit,
            },
        }
        config = apply_overrides(config, overrides)
        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self.video_processor,
            self.audio_processor,
            self.context_len,
            self.model_config,
        ) = load_model_bundle(config)
        self.config = config

        if self.image_processor is None:
            self.image_processor = ImageProcessor()
        if self.video_processor is None:
            self.video_processor = VideoProcessor()
        if self.audio_processor is None:
            self.audio_processor = AudioProcessor()

    def predict(
        self,
        image: Optional[str] = None,
        prompt: Optional[str] = None,
        top_p: float = 1.0,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
    ):
        config = copy.deepcopy(self.config)
        overrides = {
            "generation": {
                "top_p": top_p,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            }
        }
        config = apply_overrides(config, overrides)

        images = None
        if image:
            image_data = load_image(image)
            processor = self.image_processor or ImageProcessor()
            image_features = processor.preprocess(image_data, return_tensors="pt")
            image_tensor = image_features["pixel_values"] if image_features else None
            if image_tensor is not None:
                images = image_tensor.to(device=self.model.device, dtype=torch.float16)

        generation_cfg = build_generation_config(config.get("generation", {}))
        conv_mode = config.get("prompt", {}).get("conv_mode", "llava_v1")
        stop_token_ids = config.get("prompt", {}).get("stop_token_ids", [])

        _, outputs = generate_outputs(
            model=self.model,
            tokenizer=self.tokenizer,
            model_config=self.model_config,
            prompt=prompt or "",
            generation_cfg=generation_cfg,
            conv_mode=conv_mode,
            stop_token_ids=stop_token_ids,
            images=images,
        )

        media_outputs = parse_generated_media(outputs)
        assets_dir = Path(config.get("output", {}).get("assets_dir", "./assets"))
        if config.get("output", {}).get("save_images", True):
            save_images(
                media_outputs.get("images"), assets_dir / "images", prefix="output"
            )
        if config.get("output", {}).get("save_videos", True):
            save_videos(
                media_outputs.get("videos"), assets_dir / "videos", prefix="output"
            )
        if config.get("output", {}).get("save_audios", True):
            save_audios(
                media_outputs.get("audios"), assets_dir / "audios", prefix="output"
            )

        return outputs


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup(
        model_base=None,
        model_name="nextgpt-v1.5-7b",
        model_path="./checkpoints/nextgpt-v1.5-7b",
        load_8bit=False,
        load_4bit=False,
    )
    predictor.predict(
        image="./assets/bird_image.jpg",
        prompt="show me an image of a cute dog running on the grass",
    )
