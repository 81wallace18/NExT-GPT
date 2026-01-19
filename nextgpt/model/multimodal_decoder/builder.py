from .custom_ad import AudioLDMPipeline
from .custom_vd import TextToVideoSDPipeline
from .custom_sd import StableDiffusionPipeline
from diffusers import EulerDiscreteScheduler
import torch


def _apply_decoder_optimizations(decoder, config):
    dtype = getattr(config, "decoder_dtype", None)
    if dtype:
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    else:
        torch_dtype = None

    if torch_dtype is not None:
        decoder = decoder.to(dtype=torch_dtype)

    if getattr(config, "decoder_attention_slicing", False):
        decoder.enable_attention_slicing()

    if getattr(config, "decoder_xformers", False) and hasattr(
        decoder, "enable_xformers_memory_efficient_attention"
    ):
        decoder.enable_xformers_memory_efficient_attention()

    offload = getattr(config, "decoder_offload", None)
    if offload == "sequential" and hasattr(decoder, "enable_sequential_cpu_offload"):
        decoder.enable_sequential_cpu_offload()
    elif offload == "model" and hasattr(decoder, "enable_model_cpu_offload"):
        decoder.enable_model_cpu_offload()

    device = getattr(config, "decoder_device", None)
    if device:
        decoder = decoder.to(device)

    return decoder


def builder_decoder(config, decoder_modality="image"):
    if decoder_modality == "video":
        print(f"Building video decoder: {config.video_decoder}")
        video_decoder = TextToVideoSDPipeline.from_pretrained(config.video_decoder)
        video_decoder.vae.requires_grad_(False)
        video_decoder.unet.requires_grad_(False)
        video_decoder.text_encoder.requires_grad_(False)
        return _apply_decoder_optimizations(video_decoder, config)
    elif decoder_modality == "audio":
        print(f"Building audio decoder: {config.audio_decoder}")
        audio_decoder = AudioLDMPipeline.from_pretrained(config.audio_decoder)
        audio_decoder.vae.requires_grad_(False)
        audio_decoder.unet.requires_grad_(False)
        audio_decoder.text_encoder.requires_grad_(False)
        audio_decoder.vocoder.requires_grad_(False)
        return _apply_decoder_optimizations(audio_decoder, config)
    elif decoder_modality == "image":
        print(f"Building image decoder: {config.image_decoder}")
        if config.image_decoder == "stabilityai/stable-diffusion-2":
            scheduler = EulerDiscreteScheduler.from_pretrained(
                config.image_decoder, subfolder="scheduler"
            )
            image_decoder = StableDiffusionPipeline.from_pretrained(
                config.image_decoder, scheduler=scheduler
            )
        else:
            image_decoder = StableDiffusionPipeline.from_pretrained(
                config.image_decoder
            )

        image_decoder.vae.requires_grad_(False)
        image_decoder.unet.requires_grad_(False)
        image_decoder.text_encoder.requires_grad_(False)
        return _apply_decoder_optimizations(image_decoder, config)
    else:
        raise NotImplementedError(f"Decoder {decoder_modality} not implemented")
