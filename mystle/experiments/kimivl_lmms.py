"""Kimi-VL wrapper for lmms-eval."""
import torch
from typing import Optional, Union, List
from lmms_eval.api.registry import register_model
from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from PIL import Image
import sys as _sys; _sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__))); from kimivl_utils import kimi_vl_process
from tqdm import tqdm

@register_model("kimi_vl")
class KimiVLLMMS(lmms):
    def __init__(
        self,
        pretrained: str = "moonshotai/Kimi-VL-A3B-Thinking",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        device_map: Optional[str] = None,
        processor_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoProcessor
        import os

        dt = getattr(torch, dtype, torch.bfloat16)

        if device_map and device_map != "none":
            self._model = AutoModelForCausalLM.from_pretrained(
                pretrained, trust_remote_code=True, torch_dtype=dt,
                device_map=device_map, attn_implementation="eager",
            )
            self._device = "cuda:0"
        else:
            ckpt_size = sum(
                os.path.getsize(os.path.join(pretrained, f))
                for f in os.listdir(pretrained)
                if f.endswith('.safetensors') or f.endswith('.bin')
            ) if os.path.isdir(pretrained) else 0

            gpu_mem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
            needs_offload = ckpt_size > gpu_mem * 0.85

            if needs_offload:
                self._model = AutoModelForCausalLM.from_pretrained(
                    pretrained, trust_remote_code=True, torch_dtype=dt,
                    device_map="auto",
                    max_memory={0: int(gpu_mem * 0.90), "cpu": "200GiB"},
                    attn_implementation="eager",
                )
                self._device = "cuda:0"
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    pretrained, trust_remote_code=True, torch_dtype=dt,
                    device_map=device, attn_implementation="eager",
                )
                self._device = device

        proc_path = processor_path or pretrained
        if os.path.isdir(proc_path) and not os.path.exists(os.path.join(proc_path, "preprocessor_config.json")):
            proc_path = "moonshotai/Kimi-VL-A3B-Thinking"
        self.processor = AutoProcessor.from_pretrained(proc_path, trust_remote_code=True)
        self._batch_size = int(batch_size) if batch_size != "auto" else 1
        self.pretrained = pretrained

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def model(self):
        return self._model

    def tok_encode(self, string, **kwargs):
        return self.processor.tokenizer.encode(string)

    def loglikelihood(self, requests):
        raise NotImplementedError("Kimi-VL loglikelihood not implemented")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("Multi-round not implemented for Kimi-VL")

    def _to_pil_list(self, raw_visuals):
        if raw_visuals is None:
            return []
        if not isinstance(raw_visuals, (list, tuple)):
            raw_visuals = [raw_visuals]
        result = []
        for v in raw_visuals:
            if isinstance(v, (list, tuple)):
                result.extend(self._to_pil_list(v))
            elif isinstance(v, Image.Image):
                result.append(v.convert("RGB"))
            elif isinstance(v, str):
                result.append(Image.open(v).convert("RGB"))
            else:
                result.append(v)
        return result

    def generate_until(self, requests: List[Instance]) -> List[str]:
        import lmms_eval.utils as utils

        res: List[str] = []

        def _collate(x):
            toks = self.processor.tokenizer.encode(
                x[0] if isinstance(x[0], str) else str(x[0]),
                add_special_tokens=False)
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        pbar = tqdm(total=len(requests), disable=False, desc="Kimi-VL")

        for chunk in chunks:
            for args in chunk:
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = args
                ctx = contexts[0] if isinstance(contexts, (list, tuple)) else contexts
                doc = self.task_dict[task][split][doc_id]
                raw_vis = doc_to_visual(doc)
                pil_images = self._to_pil_list(raw_vis)

                gk = dict(gen_kwargs) if gen_kwargs else {}
                until = gk.pop("until", None)
                max_new = int(gk.pop("max_new_tokens", gk.pop("max_gen_toks", 1024)))
                temperature = float(gk.pop("temperature", 0.0))
                do_sample = temperature > 0

                if pil_images:
                    content = []
                    for img in pil_images:
                        content.append({"type": "image", "image": img})
                    content.append({"type": "text", "text": ctx})
                    messages = [{"role": "user", "content": content}]
                else:
                    messages = [{"role": "user", "content": [{"type": "text", "text": ctx}]}]

                text = self.processor.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                if pil_images:
                    inputs = kimi_vl_process(self.processor, text=text, images=pil_images, return_tensors="pt", padding=True)
                else:
                    inputs = kimi_vl_process(self.processor, text=text, return_tensors="pt", padding=True)
                inputs = {k: v.to(self._device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = self._model.generate(
                        **inputs, max_new_tokens=max_new,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                    )
                input_len = inputs["input_ids"].shape[-1]
                generated = output_ids[0][input_len:]
                text_out = self.processor.tokenizer.decode(generated, skip_special_tokens=True)

                if until:
                    for stop in (until if isinstance(until, list) else [until]):
                        if stop in text_out:
                            text_out = text_out[:text_out.index(stop)]

                res.append(text_out.strip())
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)
