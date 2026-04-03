"""
DeepSeek-VL2 + EleutherAI lm-evaluation-harness 适配层。

Hub 快照不含 remote code 的 .py，需使用本仓库的 ``deepseek_vl2`` 包；
且 ``HFLM._create_model`` 会向 ``from_pretrained`` 传入 ``gguf_file=None``，
部分自定义模型不接受该关键字，故在加载时临时剥除 ``None`` 的可选参数。
"""

from __future__ import annotations

import functools
from typing import Any

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM, eval_logger


def register_deepseek_vl_v2() -> None:
    """将 VL2 配置/模型注册到 Transformers Auto 映射（需在加载 HFLM 之前调用）。"""
    from transformers import AutoConfig, AutoModelForCausalLM

    from deepseek_vl2.models.modeling_deepseek_vl_v2 import (
        DeepseekVLV2Config,
        DeepseekVLV2ForCausalLM,
    )

    try:
        AutoConfig.register("deepseek_vl_v2", DeepseekVLV2Config)
        AutoModelForCausalLM.register(DeepseekVLV2Config, DeepseekVLV2ForCausalLM)
    except ValueError:
        pass


def _strip_none_hf_extras(cls: Any):
    """包装 ``from_pretrained``：不向下游 ``__init__`` 传递值为 None 的 gguf/quant 参数。"""
    orig = cls.from_pretrained

    @functools.wraps(orig)
    def wrapped(*args, **kwargs):
        if kwargs.get("gguf_file") is None:
            kwargs.pop("gguf_file", None)
        if kwargs.get("quantization_config") is None:
            kwargs.pop("quantization_config", None)
        return orig(*args, **kwargs)

    return orig, wrapped


@register_model("deepseek-vl2-hf")
class DeepSeekVL2HFLM(HFLM):
    """用于 DeepSeek-VL2 文本路径评测（无图像时走 language embedding）。"""

    def _patch_vl_top_config(self) -> None:
        """顶层 ``DeepseekVLV2Config`` 的 JSON 可能缺少 HF 通用字段，forward 会读这些属性。"""
        c = self._model.config
        defaults = {
            "use_cache": True,
            "output_attentions": False,
            "output_hidden_states": False,
            "use_return_dict": True,
        }
        for k, v in defaults.items():
            if not hasattr(c, k):
                setattr(c, k, v)

    def _create_model(self, *args, **kwargs):
        cls = self.AUTO_MODEL_CLASS
        orig, wrapped = _strip_none_hf_extras(cls)
        cls.from_pretrained = wrapped  # type: ignore[method-assign]
        try:
            super()._create_model(*args, **kwargs)
        finally:
            cls.from_pretrained = orig  # type: ignore[method-assign]
        self._patch_vl_top_config()
        eval_logger.info("DeepSeekVL2HFLM: restored AutoModelForCausalLM.from_pretrained")


# 导入即注册（evaluate.py 会先 import 本模块再调用 register_deepseek_vl_v2）
register_deepseek_vl_v2()
