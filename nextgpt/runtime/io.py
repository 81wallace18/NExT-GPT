from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Union

import scipy
from diffusers.utils import export_to_video


def ensure_dir(path: Union[str, Path]) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_images(images, output_dir: Path, prefix: str):
    if not images:
        return []
    output_dir = ensure_dir(output_dir)
    paths = []
    for idx, image in enumerate(images):
        if hasattr(image, "save"):
            output_path = output_dir / f"{prefix}_{idx}.jpg"
            image.save(output_path)
            paths.append(output_path)
    return paths


def save_videos(videos, output_dir: Path, prefix: str):
    if not videos:
        return []
    output_dir = ensure_dir(output_dir)
    paths = []
    for idx, frames in enumerate(videos):
        output_path = output_dir / f"{prefix}_{idx}.mp4"
        export_to_video(video_frames=frames, output_video_path=str(output_path))
        paths.append(output_path)
    return paths


def save_audios(audios, output_dir: Path, prefix: str, sample_rate: int = 16000):
    if not audios:
        return []
    output_dir = ensure_dir(output_dir)
    paths = []
    for idx, audio in enumerate(audios):
        output_path = output_dir / f"{prefix}_{idx}.wav"
        scipy.io.wavfile.write(str(output_path), rate=sample_rate, data=audio)
        paths.append(output_path)
    return paths


def collect_outputs(outputs: Dict) -> Dict[str, Iterable]:
    if not isinstance(outputs, dict):
        return {"images": [], "videos": [], "audios": []}

    return {
        "images": outputs.get("images") or [],
        "videos": outputs.get("videos") or [],
        "audios": outputs.get("audios") or [],
    }
