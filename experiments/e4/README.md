# E4: formal dynamic native-multiscale evaluation

E4 evaluates fixed-budget Native Front routing with Qwen2.5-VL-7B and Qwen3.5-9B. It does not alter
`FoveaConfig` with experiment-only fields and does not claim physical KV-cache compression: Native
conditions retain the full prompt cache plus `/2`, `/4`, and `/8` visual KV banks.

## Conditions

The formal and reasoning suites use an active budget of `N/16`:

| condition | behavior |
|---|---|
| `full` | complete high-resolution visual sequence |
| `lowres4` | width and height divided by four, end-to-end `N/16` input |
| `uniform4_native` | fixed 4×4 Native nodes |
| `prefill_static_top8_native` | E1 Top-8 routes once after prefill and then freezes |
| `dynamic_top8_native` | E1 Top-8 routes after prefill and every decode step |
| `dynamic_all_heads_native` | all Full-Attention heads route dynamically |

The compression suite runs `lowres2`, `uniform2_native`, `prefill_static2_top8_native`, and
`dynamic2_top8_native` at `N/4`. Split–Merge updates preserve the initialized node count exactly.
Prefill-static conditions also stop collecting visual routing signals during decode.

The formal suite uses original task prompts. The reasoning suite uses a fixed 400-example subset and
requests a natural analysis of at most 200 words followed by `<answer>...</answer>`. Thinking remains
disabled. Because TokenFovea leaves prompt prefill unchanged, all high-resolution Native conditions
must produce the same first output token as `full`; routing affects later decode forwards.

## Benchmarks

Built-in lmms-eval tasks: HR-Bench-8K, XLRS-Bench-lite, V*Bench, MMStar, ChartQA, and TextVQA.

Local task adapters under `tasks/` add:

- VisualProbe Easy/Medium/Hard: all 515 validation examples, deterministic Acc@1 rather than Avg@32.
- FineRS-QA: answerable MVQA/OVQA rows only; segmentation-only rows are excluded.
- HRScene: `realworld_combined/testmini`, 1000 examples; hidden test is not run.

VisualProbe snapshots and FineRS images are separate from their annotation tables. Set these variables
when their stored paths are not directly readable:

```bash
export VISUALPROBE_ROOT=/data/VisualProbe
export FINERS4K_IMAGE_ROOT=/data/Finers-4k/images
```

The processed high-resolution grid is aligned to eight and capped at 4096 LLM visual tokens. The
configured `max_pixels` values are `4096*28^2` for Qwen2.5-VL and `4096*32^2` for Qwen3.5.

## Run

E1 Top-8 files are required by Top-8 conditions:

```text
outputs/e1/qwen25/head_selection_top8.json
outputs/e1/qwen35/head_selection_top8.json
```

Commands:

```bash
python -m experiments.e4 prepare --config experiments/e4/configs/default.yaml

python -m experiments.e4 run --model qwen25 --suite formal
python -m experiments.e4 run --model qwen25 --suite reasoning
python -m experiments.e4 run --model qwen25 --suite compression

python -m experiments.e4 run --model qwen25 --suite formal \
  --condition dynamic_top8_native --task vstar_bench

python -m experiments.e4 analyze --model qwen25
python -m experiments.e4 report --model qwen25
```

`prepare` also downloads the three VisualProbe image snapshots and FineRS `all_images.zip` (roughly
18 GB) and extracts it. Existing files and the extraction marker are reused.

Use `torchrun` for process sharding in the same way as E2/E3. Per-rank JSONL files are merged by sample
ID, completed samples are skipped, and a condition-level `results.json` is written only after every
task in that suite is complete.

Outputs are stored under `outputs/e4/<model>/<suite>/<condition>/`. The model root contains the run
manifest, `summary.json`, `summary.csv`, and `report.html`. Route traces include active fronts,
Split–Merge counts, scale distributions, churn, and top node scores.

## Local verification

The current non-CUDA machine can run only CPU checks:

```bash
pytest tests/experiments/e4 tests/unit tests/integration
python scripts/smoke_e4.py
python -m compileall src/tokenfovea experiments/e4
```

Formal inference intentionally fails before model loading when CUDA is unavailable.
