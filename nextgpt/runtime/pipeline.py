from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from nextgpt.constants import IMAGE_TOKEN_INDEX
from nextgpt.conversation import SeparatorStyle, conv_templates
from nextgpt.mm_utils import tokenizer_multiple_token


@dataclass
class GenerationConfig:
    top_k: int = 1
    top_p: float = 1.0
    temperature: float = 0.2
    max_new_tokens: int = 512
    do_sample: bool = True
    use_cache: bool = False
    output_hidden_states: bool = True
    guidance_scale_for_img: float = 7.5
    num_inference_steps_for_img: int = 50
    guidance_scale_for_vid: float = 7.5
    num_inference_steps_for_vid: int = 50
    height: int = 320
    width: int = 576
    num_frames: int = 16
    guidance_scale_for_aud: float = 7.5
    num_inference_steps_for_aud: int = 50
    audio_length_in_s: float = 5.0


class StopTokensCriteria(StoppingCriteria):
    def __init__(self, stop_ids, encounters: int = 1):
        super().__init__()
        if stop_ids is None:
            stop_ids = []
        normalized = []
        for stop_id in stop_ids:
            if isinstance(stop_id, int):
                normalized.append([stop_id])
            else:
                normalized.append(stop_id)
        self.stop_ids = normalized
        self.encounters = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop_id in self.stop_ids:
            stop_tensor = torch.tensor(stop_id).to(input_ids[0].device)
            indices = torch.where(stop_tensor[0] == input_ids)
            for idx in indices:
                if len(idx) > 0:
                    if torch.all(
                        input_ids[0][idx : idx + len(stop_tensor)] == stop_tensor
                    ):
                        stop_count += 1
        return stop_count >= self.encounters


def build_generation_config(raw: Dict[str, Any]) -> GenerationConfig:
    return GenerationConfig(**raw)


def build_prompt(prompt: str, conv_mode: str) -> str:
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def prepare_inputs(prompt: str, tokenizer, conv_mode: str):
    formatted = build_prompt(prompt, conv_mode)
    input_ids = tokenizer_multiple_token(
        formatted,
        tokenizer,
        IMAGE_TOKEN_INDEX,
    )
    input_ids = torch.as_tensor(input_ids, dtype=torch.long).unsqueeze(0)
    return formatted, input_ids


def build_signal_indices(tokenizer, model_config):
    image_tokens = [
        tokenizer(f"<image_{i:02d}>").input_ids
        for i in range(model_config.n_img_tokens)
    ]
    video_tokens = [
        tokenizer(f"<video_{i:02d}>").input_ids
        for i in range(model_config.n_vid_tokens)
    ]
    audio_tokens = [
        tokenizer(f"<audio_{i:02d}>").input_ids
        for i in range(model_config.n_aud_tokens)
    ]
    return image_tokens, video_tokens, audio_tokens


def generate_outputs(
    model,
    tokenizer,
    model_config,
    prompt: str,
    generation_cfg: GenerationConfig,
    conv_mode: str,
    stop_token_ids,
    images: Optional[torch.Tensor] = None,
    videos: Optional[torch.Tensor] = None,
    audios: Optional[torch.Tensor] = None,
):
    formatted, input_ids = prepare_inputs(prompt, tokenizer, conv_mode)
    input_ids = input_ids.to(model.device)

    stop_criteria = (
        StoppingCriteriaList([StopTokensCriteria(stop_ids=stop_token_ids)])
        if stop_token_ids
        else None
    )
    image_tokens, video_tokens, audio_tokens = build_signal_indices(
        tokenizer, model_config
    )

    outputs = model.generate(
        input_ids=input_ids,
        images=images,
        videos=videos,
        audios=audios,
        image_signal_token_indices=image_tokens,
        video_signal_token_indices=video_tokens,
        audio_signal_token_indices=audio_tokens,
        stopping_criteria=stop_criteria,
        **generation_cfg.__dict__,
    )
    return formatted, outputs
