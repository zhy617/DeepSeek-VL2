"""Kimi-VL evaluation script for lmms-eval (MM) and lm-evaluation-harness (Text)."""
import argparse, json, os, sys, datetime

ORIGINAL_MODEL = "moonshotai/Kimi-VL-A3B-Thinking"


def _register_kimi_vl():
    import mystle.experiments.kimivl_lmms  # noqa: F401
    from lmms_eval.models import MODEL_REGISTRY_V2
    from lmms_eval.models.registry_v2 import ModelManifest
    MODEL_REGISTRY_V2.register_manifest(
        ModelManifest(
            model_id="kimi_vl",
            simple_class_path="mystle.experiments.kimivl_lmms.KimiVLLMMS",
        ),
        overwrite=True,
    )


def run_mm_eval(args):
    _register_kimi_vl()
    from lmms_eval.evaluator import simple_evaluate

    os.makedirs(args.output_dir, exist_ok=True)

    processor_path = ORIGINAL_MODEL
    if os.path.isdir(args.model_path):
        pp_check = os.path.join(args.model_path, "preprocessor_config.json")
        if os.path.exists(pp_check):
            processor_path = args.model_path

    model_args = f"pretrained={args.model_path},dtype=bfloat16,processor_path={processor_path}"
    task_list = ["mme", "ocrbench", "scienceqa_img", "ai2d", "pope", "realworldqa"]

    results = simple_evaluate(
        model="kimi_vl",
        model_args=model_args,
        tasks=task_list,
        batch_size=1,
    )

    out_path = os.path.join(args.output_dir, "lmms_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results.get("results", results), f, indent=2, default=str)

    meta = {
        "run_id": os.path.basename(args.output_dir),
        "pretrained": args.model_path,
        "tasks": task_list,
        "batch_size": 1,
        "model_type": "kimi_vl",
        "framework": "lmms-eval",
        "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"MM eval results saved to {out_path}")
    return results


def run_text_eval(args):
    from lm_eval import simple_evaluate

    os.makedirs(args.output_dir, exist_ok=True)
    model_args = f"pretrained={args.model_path},dtype=bfloat16,trust_remote_code=True"
    task_list = ["arc_easy", "winogrande", "piqa"]

    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=task_list,
        batch_size=4,
    )

    out_path = os.path.join(args.output_dir, "lm_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results.get("results", results), f, indent=2, default=str)

    meta = {
        "run_id": os.path.basename(args.output_dir),
        "pretrained": args.model_path,
        "tasks": task_list,
        "batch_size": 4,
        "model_type": "kimi_vl_hf",
        "framework": "lm-evaluation-harness",
        "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Text eval results saved to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["mm", "text"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.mode == "mm":
        run_mm_eval(args)
    else:
        run_text_eval(args)
