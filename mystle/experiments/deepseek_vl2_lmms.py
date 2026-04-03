"""
DeepSeek-VL2 与 lmms-eval 的适配（多模态 generate_until）。

需在 ``PYTHONPATH`` 中包含本仓库根目录以加载 ``deepseek_vl2``；
通过 ``mystle.experiments.evaluate_mm`` 在运行时注册模型 ``deepseek_vl2``。
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

def _register_hf_classes() -> None:
    from transformers import AutoConfig, AutoModelForCausalLM as AMC

    from deepseek_vl2.models.modeling_deepseek_vl_v2 import (
        DeepseekVLV2Config,
        DeepseekVLV2ForCausalLM,
    )

    try:
        AutoConfig.register("deepseek_vl_v2", DeepseekVLV2Config)
        AMC.register(DeepseekVLV2Config, DeepseekVLV2ForCausalLM)
    except ValueError:
        pass


def _strip_none_hf_extras(cls: Any):
    orig = cls.from_pretrained

    @functools.wraps(orig)
    def wrapped(*args, **kwargs):
        if kwargs.get("gguf_file") is None:
            kwargs.pop("gguf_file", None)
        if kwargs.get("quantization_config") is None:
            kwargs.pop("quantization_config", None)
        return orig(*args, **kwargs)

    return orig, wrapped


def _patch_top_config(model: Any) -> None:
    c = model.config
    defaults = {
        "use_cache": True,
        "output_attentions": False,
        "output_hidden_states": False,
        "use_return_dict": True,
    }
    for k, v in defaults.items():
        if not hasattr(c, k):
            setattr(c, k, v)


def _flatten_visuals(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        if len(v) == 1 and isinstance(v[0], list):
            return v[0]
        out: List[Any] = []
        for x in v:
            if x is None:
                continue
            if isinstance(x, list):
                out.extend(x)
            else:
                out.append(x)
        return out
    return [v]


def _to_rgb_list(visuals: List[Any]) -> List[Image.Image]:
    out: List[Image.Image] = []
    for v in visuals:
        if isinstance(v, Image.Image):
            out.append(v.convert("RGB"))
        elif isinstance(v, str) and Path(v).exists():
            out.append(Image.open(v).convert("RGB"))
        else:
            eval_logger.warning(f"Unsupported visual type {type(v)}, skipping")
    return out


def _align_image_tokens(user_text: str, pil_images: List[Image.Image]) -> str:
    n = len(pil_images)
    if n == 0:
        return user_text
    tag = "<image>"
    c = user_text.count(tag)
    if c == n:
        return user_text
    if c == 0:
        return (tag * n) + "\n" + user_text.strip()
    eval_logger.warning(
        f"Prompt has {c} {tag} but {n} images; prefixing {n} tokens and stripping extras in prompt."
    )
    stripped = user_text.replace(tag, "").strip()
    return (tag * n) + "\n" + stripped


@register_model("deepseek_vl2")
class DeepseekVL2LMMS(lmms):
    """DeepSeek-VL2 多模态评测（lmms-eval）。"""

    def __init__(
        self,
        pretrained: str = "deepseek-ai/deepseek-vl2-small",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        dtype: str = "bfloat16",
        chunk_size: int = 512,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            eval_logger.warning(f"DeepseekVL2LMMS: ignoring unexpected kwargs: {kwargs}")

        _register_hf_classes()
        self._device = torch.device(device if device else "cuda:0")
        self.batch_size_per_gpu = int(batch_size)
        if self.batch_size_per_gpu != 1:
            eval_logger.warning(
                "DeepseekVL2LMMS: batch_size>1 未充分验证，将按 batch_size=1 逐条推理。"
            )
            self.batch_size_per_gpu = 1

        self.chunk_size = chunk_size

        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
            dtype, torch.bfloat16
        )

        from deepseek_vl2.models import DeepseekVLV2Processor
        from deepseek_vl2.models.modeling_deepseek_vl_v2 import DeepseekVLV2ForCausalLM

        cls = DeepseekVLV2ForCausalLM
        orig, wrapped = _strip_none_hf_extras(cls)
        cls.from_pretrained = wrapped  # type: ignore[method-assign]
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                trust_remote_code=trust_remote_code,
                torch_dtype=dt,
            )
        finally:
            cls.from_pretrained = orig  # type: ignore[method-assign]

        self._model = self._model.to(self._device).eval()
        _patch_top_config(self._model)

        self.processor = DeepseekVLV2Processor.from_pretrained(pretrained)
        self._tokenizer = self.processor.tokenizer
        self._config = self._model.config

        self._rank = 0
        self._world_size = 1
        self.use_cache = True

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("DeepseekVL2LMMS: loglikelihood 未实现")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("DeepseekVL2LMMS: multi-round 未实现")

    def _generate_one(self, user_text: str, pil_images: List[Image.Image], gen_kwargs: dict) -> str:
        user_text = _align_image_tokens(user_text, pil_images)
        conversations = [
            {"role": "<|User|>", "content": user_text},
            {"role": "<|Assistant|>", "content": ""},
        ]
        prepare_inputs = self.processor(
            conversations=conversations,
            images=pil_images,
            force_batchify=True,
            inference_mode=True,
            system_prompt="",
        ).to(self._device)

        gk = dict(gen_kwargs) if gen_kwargs else {}
        until = gk.pop("until", None)
        if isinstance(until, str):
            until_list = [until]
        elif isinstance(until, list):
            until_list = until
        else:
            until_list = [self.tokenizer.decode([self.eot_token_id])]

        max_new_tokens = int(gk.pop("max_new_tokens", 128))
        temperature = float(gk.pop("temperature", 0.0))
        top_p = gk.pop("top_p", 1.0)
        do_sample = bool(gk.pop("do_sample", temperature > 0))
        num_beams = int(gk.pop("num_beams", 1))
        if gk:
            eval_logger.debug(f"Unused generation kwargs: {gk}")

        with torch.inference_mode():
            if self.chunk_size and self.chunk_size > 0:
                inputs_embeds, past_key_values = self._model.incremental_prefilling(
                    input_ids=prepare_inputs.input_ids,
                    images=prepare_inputs.images,
                    images_seq_mask=prepare_inputs.images_seq_mask,
                    images_spatial_crop=prepare_inputs.images_spatial_crop,
                    attention_mask=prepare_inputs.attention_mask,
                    chunk_size=self.chunk_size,
                )
            else:
                inputs_embeds = self._model.prepare_inputs_embeds(**prepare_inputs)
                past_key_values = None

            gen_cfg = {
                "inputs_embeds": inputs_embeds,
                "input_ids": prepare_inputs.input_ids,
                "images": prepare_inputs.images,
                "images_seq_mask": prepare_inputs.images_seq_mask,
                "images_spatial_crop": prepare_inputs.images_spatial_crop,
                "attention_mask": prepare_inputs.attention_mask,
                "past_key_values": past_key_values,
                "pad_token_id": self.tokenizer.pad_token_id,
                "bos_token_id": self.tokenizer.bos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "use_cache": True,
            }
            if do_sample:
                gen_cfg["temperature"] = temperature
                gen_cfg["top_p"] = top_p
            if num_beams > 1:
                gen_cfg["num_beams"] = num_beams

            outputs = self._model.generate(**gen_cfg)

        in_len = prepare_inputs.input_ids.shape[1]
        new_tokens = outputs[0, in_len:]
        text = self.tokenizer.decode(new_tokens.cpu().tolist(), skip_special_tokens=True)

        stop_pos = len(text)
        for term in until_list:
            if term and term in text:
                stop_pos = min(stop_pos, text.index(term))
        text = text[:stop_pos].strip()
        return text

    def generate_until(self, requests: List[Instance]) -> List[str]:
        import lmms_eval.utils as utils

        res: List[str] = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0], add_special_tokens=False)
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="DeepSeek-VL2")

        for chunk in chunks:
            for args in chunk:
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = args
                ctx = contexts[0] if isinstance(contexts, (list, tuple)) else contexts
                doc = self.task_dict[task][split][doc_id]
                raw_vis = doc_to_visual(doc)
                flat = _flatten_visuals(raw_vis)
                pil_images = _to_rgb_list(flat)
                gk = dict(gen_kwargs) if gen_kwargs else {}
                out = self._generate_one(ctx, pil_images, gk)
                res.append(out)
                self.cache_hook.add_partial("generate_until", (ctx, gk), out)
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)


# 导入本模块时注册到旧 registry（部分测试用）；主路径由 evaluate_mm 写 ModelRegistryV2
