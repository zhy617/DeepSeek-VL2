"""Shared utilities for Kimi-VL experiments.

Provides a processor wrapper that bypasses the broken _merge_kwargs
in transformers 4.44.2 with KimiVLProcessor.
"""
import torch
from typing import List, Optional, Union
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature


def kimi_vl_process(
    processor,
    text: Union[str, List[str]],
    images: Optional[List[Image.Image]] = None,
    return_tensors: str = "pt",
    padding: bool = False,
) -> BatchFeature:
    """Process text and images for Kimi-VL, bypassing broken _merge_kwargs.

    Replicates the logic of KimiVLProcessor.__call__ without calling
    self._merge_kwargs which fails on transformers >= 4.44.
    """
    if isinstance(text, str):
        text = [text]

    image_inputs = {}
    if images is not None:
        image_inputs = processor.image_processor(images, return_tensors=return_tensors)
        image_grid_hws = image_inputs["image_grid_hws"]

        image_token = processor.image_token
        merge_length = (
            processor.image_processor.merge_kernel_size[0]
            * processor.image_processor.merge_kernel_size[1]
        )
        index = 0
        for i in range(len(text)):
            while image_token in text[i]:
                n_tokens = image_grid_hws[index].prod() // merge_length
                text[i] = text[i].replace(
                    image_token,
                    "<|placeholder|>" * int(n_tokens),
                    1,
                )
                index += 1
            text[i] = text[i].replace("<|placeholder|>", image_token)

    text_inputs = processor.tokenizer(
        text, return_tensors=return_tensors, padding=padding
    )

    return BatchFeature(data={**text_inputs, **image_inputs})


def prepare_kimi_vl_inputs(
    processor,
    messages: list,
    images: Optional[List[Image.Image]] = None,
    return_tensors: str = "pt",
    device: Optional[str] = None,
) -> dict:
    """High-level: apply chat template + process, return model-ready inputs."""
    text = processor.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = kimi_vl_process(
        processor, text=text, images=images, return_tensors=return_tensors
    )
    result = dict(inputs)
    if device:
        result = {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in result.items()
        }
    return result
