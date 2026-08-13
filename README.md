# TokenFovea

Fixed-budget multi-scale visual KV routing for Qwen2.5-VL and Qwen3.5.

TokenFovea keeps prompt prefill unchanged, builds one pre-RoPE visual KV pyramid per routed decoder layer, and substitutes a fixed-size spatial front for the fine visual tokens in decode-time attention. The original Hugging Face cache is retained for generation bookkeeping and is not compressed.

This repository currently contains the routing implementation and its core tests. Experiment-specific probe and evaluation pipelines are intentionally outside the core package.

## Structure

- `topology.py`: visual-token quadtrees and their device tensor representation.
- `pyramid.py`: level-wise construction and gathering of per-layer visual KV pyramids.
- `router.py`: device-resident Split-Merge routing that preserves the initialized front size.
- `session.py`: state and KV composition for one prompt/decode session.
- `integrations/qwen/`: shared attention utilities, model-specific forwards, and patch installation.
- `integrations/lmms_eval.py`: lmms-eval model adapters.

The decode path keeps routing scores and state on the accelerator. Normal attention uses PyTorch SDPA; only the small visual attention distribution needed by the router is computed separately. Tree construction happens once on the CPU when a prompt starts.

## Run

```bash
tokenfovea \
  --model tokenfovea_qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,batch_size=1,attn_implementation=sdpa,fovea_budget=512,fovea_mode=dynamic \
  --tasks textvqa_val \
  --batch_size 1 \
  --output_path outputs/textvqa
```

Use `tokenfovea_qwen3_5` with `Qwen/Qwen3.5-9B` for Qwen3.5.

Main arguments:

- `fovea_mode=dynamic|uniform|full`
- `fovea_budget`
- `fovea_position_mode=native_center|text_anchor|no_rope|post_rope_pool`
- `fovea_pooling_mode=kv|hidden`
- `fovea_signal_selection=/path/to/head_selection.json` (optional; all heads are used when omitted)
- `fovea_signal_aggregation=mean|max`
- `fovea_anchor_window` (used only by `text_anchor`)
- `fovea_update_interval`
- `fovea_max_swaps`
- `fovea_epsilon`
- `fovea_attention_ema`
- `fovea_score_mode=mass|density`
- `fovea_route_after_prefill=true|false`

`fovea_budget` is a target number of active visual nodes, not a pixel limit. The actual front size is the closest size reachable by the ragged quadtree and cannot exceed the number of visual tokens produced by the model processor. Image resolution is controlled separately by the lmms-eval model arguments `min_pixels` and `max_pixels`; `total_pixels` applies to video preprocessing, which TokenFovea does not currently support. For multiple images, `max_pixels` is applied to each image independently.

Current constraints for routed modes: image input only, batch size one, `num_beams=1`, and `use_cache=true`. The full multi-scale KV pyramid still occupies memory proportional to the original visual-token count. If `fovea_budget` is greater than or equal to the processed visual-token count, the front contains all leaf nodes and no spatial compression occurs.

## Test

```bash
pytest tests
```

The routing-only dynamic-resolution smoke test can be run without loading a VLM:

```bash
python scripts/smoke_dynamic_resolution.py test.png
```

## E1 Head discovery

The no-ROI visual routing Head probe is implemented separately under `experiments/e1`. It measures
full-context visual attention, visual-only HybridKV dynamics, and null-calibrated 3×3 GazeScore,
then exports Top-4/8/16 `head_selection.json` candidates and an HTML report. See
[`experiments/e1/README.md`](experiments/e1/README.md) for NVIDIA setup and commands.

## E2 coarse visual representation

The independent E2 module compares uniform and random multiscale visual pooling with token-matched native
low-resolution inputs on four lmms-eval Lite tasks. See [`experiments/e2/README.md`](experiments/e2/README.md).
