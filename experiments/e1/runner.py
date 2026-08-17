from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm  # type: ignore[import-untyped]

from experiments.distributed import distributed_context, merge_rank_jsonl

from .config import E1Config, ModelSpec
from .data import prepare_data, read_jsonl
from .probe import E1AttentionProbe, checkpoint_sample_ids, merge_probe_checkpoints


def run_probe(config: E1Config, model_name: str, *, pass_name: str = "scan") -> Path:
    distributed = distributed_context()
    if model_name not in config.models:
        raise ValueError(f"unknown model {model_name!r}; choose from {sorted(config.models)}")
    if pass_name not in {"scan", "visualize"}:
        raise ValueError("pass_name must be 'scan' or 'visualize'")
    if distributed.is_main:
        prepare_data(config)
    distributed.barrier()
    spec = config.models[model_name]
    output_dir = config.output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(spec, model_name)
    if pass_name == "scan":
        _scan(config, model_name, spec, model, processor, output_dir)
    else:
        _visualize(config, model_name, model, processor, output_dir)
    return output_dir


def run_all(config: E1Config, model_name: str) -> Path:
    """Run prepare, scan, analysis, visualization, and reporting with one model load."""

    distributed = distributed_context()
    if model_name not in config.models:
        raise ValueError(f"unknown model {model_name!r}; choose from {sorted(config.models)}")
    if distributed.is_main:
        prepare_data(config)
    distributed.barrier()
    spec = config.models[model_name]
    output_dir = config.output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(spec, model_name)
    _scan(config, model_name, spec, model, processor, output_dir)
    from .analysis import analyze
    from .report import build_report

    if distributed.is_main:
        analyze(config, model_name)
    distributed.barrier()
    _visualize(config, model_name, model, processor, output_dir)
    distributed.barrier()
    report_path = output_dir / "report.html"
    if distributed.is_main:
        report_path = build_report(config, model_name)
    distributed.barrier()
    return report_path


def load_model(spec: ModelSpec, model_name: str) -> tuple[Any, Any]:
    distributed = distributed_context()
    try:
        from transformers import AutoProcessor
        if model_name == "qwen25":
            from transformers import Qwen2_5_VLForConditionalGeneration

            model_class = Qwen2_5_VLForConditionalGeneration
        else:
            from transformers import Qwen3_5ForConditionalGeneration

            model_class = Qwen3_5ForConditionalGeneration
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "E1 requires a Transformers build supporting Qwen2.5-VL and Qwen3.5; "
            "install the NVIDIA environment described in experiments/e1/README.md"
        ) from error
    model = model_class.from_pretrained(
        spec.pretrained,
        torch_dtype=torch.bfloat16,
        device_map=distributed.device if distributed.enabled else "auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        spec.pretrained,
        min_pixels=spec.min_pixels,
        max_pixels=spec.max_pixels,
        trust_remote_code=True,
        local_files_only=True,
    )
    return model, processor


def _scan(
    config: E1Config,
    model_name: str,
    spec: ModelSpec,
    model: Any,
    processor: Any,
    output_dir: Path,
) -> None:
    distributed = distributed_context()
    checkpoint_path = output_dir / "probe_checkpoint.json"
    if distributed.enabled:
        distributed_root = output_dir / ".distributed"
        rank_output_dir = distributed_root / f"rank{distributed.rank}"
        if distributed.is_main:
            distributed_root.mkdir(parents=True, exist_ok=True)
            existing = list(distributed_root.glob("rank*/probe_checkpoint.json"))
            if checkpoint_path.exists() and not existing:
                seed = distributed_root / "rank0" / "probe_checkpoint.json"
                seed.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(checkpoint_path, seed)
        distributed.barrier()
        checkpoint_paths = sorted(distributed_root.glob("rank*/probe_checkpoint.json"))
        completed_ids = checkpoint_sample_ids(checkpoint_paths)
    else:
        rank_output_dir = output_dir
        checkpoint_paths = [checkpoint_path]
        completed_ids = checkpoint_sample_ids(checkpoint_paths)
    probe = E1AttentionProbe(
        model,
        rank_output_dir,
        top_fraction=config.hybrid_top_fraction,
        checkpoint=True,
    )
    try:
        natural = list(read_jsonl(config.data_dir / "natural.jsonl"))
        pending_natural = [record for record in natural if record["id"] not in completed_ids]
        local_natural = distributed.shard(pending_natural)
        for record in tqdm(
            local_natural,
            total=len(local_natural),
            desc=(
                f"{model_name} natural probe "
                f"rank {distributed.rank}/{distributed.world_size}"
            ),
        ):
            image = _load_image(config, record)
            inputs = _prepare_inputs(model_name, model, processor, image, record["prompt"])
            probe.begin_sample(record["id"], "natural", inputs["input_ids"], inputs["image_grid_thw"])
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=config.generation_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )
            generated = output[0, inputs["input_ids"].shape[-1] :]
            text = processor.decode(generated, skip_special_tokens=True)
            probe.end_sample(text)

        controlled = list(read_jsonl(config.data_dir / "controlled.jsonl"))
        sample_ids = [
            f"{record['id']}-panel-{panel}"
            for record in controlled
            for panel in range(len(record["prompts"]))
        ] + [f"{record['id']}-null" for record in controlled]
        pending_controlled = [sample_id for sample_id in sample_ids if sample_id not in completed_ids]
        local_controlled = set(distributed.shard(pending_controlled))
        progress = tqdm(
            total=len(local_controlled),
            desc=(
                f"{model_name} gaze probe "
                f"rank {distributed.rank}/{distributed.world_size}"
            ),
        )
        for record in controlled:
            record_ids = {
                *(f"{record['id']}-panel-{panel}" for panel in range(len(record["prompts"]))),
                f"{record['id']}-null",
            }
            if not record_ids.intersection(local_controlled):
                continue
            image = _load_image(config, record)
            for panel, prompt in enumerate(record["prompts"]):
                sample_id = f"{record['id']}-panel-{panel}"
                if sample_id not in local_controlled:
                    continue
                inputs = _prepare_inputs(model_name, model, processor, image, prompt)
                probe.begin_sample(
                    sample_id,
                    "gaze",
                    inputs["input_ids"],
                    inputs["image_grid_thw"],
                    target_panel=panel,
                )
                with torch.inference_mode():
                    model(**inputs, use_cache=False, return_dict=True)
                probe.end_sample()
                progress.update()
            null_id = f"{record['id']}-null"
            if null_id not in local_controlled:
                continue
            inputs = _prepare_inputs(model_name, model, processor, image, record["null_prompt"])
            probe.begin_sample(
                null_id, "null", inputs["input_ids"], inputs["image_grid_thw"]
            )
            with torch.inference_mode():
                model(**inputs, use_cache=False, return_dict=True)
            probe.end_sample()
            progress.update()
        progress.close()
        probe.save()
    finally:
        probe.handle.remove()
    distributed.barrier()
    if distributed.is_main:
        if distributed.enabled:
            checkpoint_paths = sorted(
                (output_dir / ".distributed").glob("rank*/probe_checkpoint.json")
            )
            merge_probe_checkpoints(checkpoint_paths, checkpoint_path)
            summary_probe = E1AttentionProbe(
                model,
                output_dir,
                top_fraction=config.hybrid_top_fraction,
                checkpoint=True,
            )
            try:
                summary_probe.save()
            finally:
                summary_probe.handle.remove()
        metadata_path = output_dir / "probe_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "model": spec.pretrained,
                "model_alias": model_name,
                "seed": config.seed,
                "generation_tokens": config.generation_tokens,
                "natural_image_count": len(natural),
                "controlled_image_count": len(controlled),
            }
        )
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    distributed.barrier()


def _visualize(
    config: E1Config,
    model_name: str,
    model: Any,
    processor: Any,
    output_dir: Path,
) -> None:
    distributed = distributed_context()
    selection_path = output_dir / "head_selection_top16.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"run analyze before the visualization pass: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    heads = {(int(item["layer"]), int(item["head"])) for item in selection["selected_heads"]}
    trace_dir = output_dir / "visualization"
    trace_path = trace_dir / "attention_traces.jsonl"
    trace_dir.mkdir(parents=True, exist_ok=True)
    if distributed.is_main:
        if trace_path.exists():
            trace_path.unlink()
        for shard in trace_dir.glob("attention_traces.rank*.jsonl"):
            shard.unlink()
    distributed.barrier()
    write_trace_path = (
        distributed.rank_path(trace_path) if distributed.enabled else trace_path
    )
    probe = E1AttentionProbe(
        model,
        trace_dir,
        top_fraction=config.hybrid_top_fraction,
        trace_heads=heads,
        trace_path=write_trace_path,
    )
    try:
        natural = list(read_jsonl(config.data_dir / "natural.jsonl"))[: config.visualization_natural]
        local_natural = distributed.shard(natural)
        for record in tqdm(
            local_natural,
            desc=(
                f"{model_name} natural visualization "
                f"rank {distributed.rank}/{distributed.world_size}"
            ),
        ):
            image = _load_image(config, record)
            inputs = _prepare_inputs(model_name, model, processor, image, record["prompt"])
            probe.begin_sample(record["id"], "trace", inputs["input_ids"], inputs["image_grid_thw"])
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=config.generation_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )
            generated = output[0, inputs["input_ids"].shape[-1] :]
            probe.end_sample(processor.decode(generated, skip_special_tokens=True))

        controlled = list(read_jsonl(config.data_dir / "controlled.jsonl"))[: config.visualization_controlled]
        controlled_items = [
            (record, panel, prompt)
            for record in controlled
            for panel, prompt in enumerate(record["prompts"])
        ]
        local_controlled = distributed.shard(controlled_items)
        for record, panel, prompt in tqdm(
            local_controlled,
            desc=(
                f"{model_name} gaze visualization "
                f"rank {distributed.rank}/{distributed.world_size}"
            ),
        ):
            image = _load_image(config, record)
            inputs = _prepare_inputs(model_name, model, processor, image, prompt)
            probe.begin_sample(
                f"{record['id']}-panel-{panel}",
                "gaze",
                inputs["input_ids"],
                inputs["image_grid_thw"],
                target_panel=panel,
            )
            with torch.inference_mode():
                model(**inputs, use_cache=False, return_dict=True)
            probe.end_sample()
    finally:
        probe.handle.remove()
    distributed.barrier()
    if distributed.is_main and distributed.enabled:
        merge_rank_jsonl(trace_path, key="sample_id")
    distributed.barrier()


def _prepare_inputs(model_name: str, model: Any, processor: Any, image: Image.Image, prompt: str) -> Any:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if model_name == "qwen35":
        template_kwargs["enable_thinking"] = False
    text = processor.apply_chat_template(messages, **template_kwargs)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    device = _input_device(model)
    return inputs.to(device)


def _input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None and torch.device(device).type != "meta":
        return torch.device(device)
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("could not determine model input device")


def _load_image(config: E1Config, record: dict[str, Any]) -> Image.Image:
    return Image.open(config.data_dir / record["image"]).convert("RGB")
