from __future__ import annotations

from typing import Dict, List

import torch


def _filter_tensors(items: List):
    return [item for item in items if not torch.is_tensor(item)]


def parse_generated_media(outputs: Dict) -> Dict[str, List]:
    if not isinstance(outputs, dict):
        return {"images": [], "videos": [], "audios": []}

    images = outputs.get("images") or []
    videos = outputs.get("videos") or []
    audios = outputs.get("audios") or []

    parsed_images: List = []
    for item in _filter_tensors(images):
        if isinstance(item, list):
            parsed_images.extend([img for img in item if hasattr(img, "save")])
        elif hasattr(item, "save"):
            parsed_images.append(item)

    parsed_videos = []
    for item in _filter_tensors(videos):
        if isinstance(item, list):
            parsed_videos.append(item)
        else:
            parsed_videos.append(item)

    parsed_audios = _filter_tensors(audios)

    return {
        "images": parsed_images,
        "videos": parsed_videos,
        "audios": parsed_audios,
    }
