# E4: formal dynamic native-multiscale evaluation

E4 evaluates fixed-budget Native Front routing with Qwen2.5-VL-7B and Qwen3.5-9B. It does not alter
`FoveaConfig` with experiment-only fields and does not claim physical KV-cache compression: Native
conditions retain the full prompt cache plus `/2`, `/4`, and `/8` visual KV banks.

## Conditions

`compression_ratio` is the requested **token-count ratio** `N/B`, not a spatial
side-length divisor. Experiments A (`formal`) and B (`reasoning`) default to `8`,
so each sample targets `B ≈ N/8`. The condition names include the configured ratio:

| condition | behavior |
|---|---|
| `full` | complete high-resolution visual sequence |
| `lowres8` | aspect-preserving LowRes input at the shared `B` budget |
| `uniform8_native` | fixed spatially uniform Native front of exactly `B` nodes |
| `prefill_static8_top8_native` | Top-8 prefill routing at `B`, then freezes |
| `dynamic8_top8_native` | Top-8 prefill routing followed by per-token updates |
| `dynamic8_all_heads_native` | same dynamic schedule using all Full-Attention heads |

Experiment C (`compression`) sweeps `compression_ratios`, which defaults to
`[2, 4, 6, 8, 16]`. At each ratio it runs LowRes, Uniform Native, Prefill Static
Top-8, and Dynamic Top-8, for 20 conditions in total. Prefill-static conditions
stop collecting visual routing signals during decode.

Every condition starts from the same high-resolution image plan: the LLM-token grid is aligned to
eight and capped before either the control or an experimental condition is derived. For a requested
ratio `r`, E4 jointly searches aspect-preserving integer LowRes grids close to `N/r` whose total token
count is also an exactly reachable Native-front budget. LowRes and every Native condition then use
that identical actual budget `B`; `full` uses the shared aligned `N`. This avoids condition-specific
high-resolution alignment. Since integer grids cannot always hit `N/r` exactly, samples record the
configured ratio, theoretical budget, actual budget, achieved `N/B`, and retained fraction. A ratio
such as `6` does not require a 6-token tree node: the Native front mixes its existing local scales
while preserving the global node count exactly.

Experiments A and C use the same full formal dataset indices and original task prompts. Experiment B
uses a fixed 400-example mechanism subset and requests a natural analysis of at most 200 words followed
by `<answer>...</answer>`. Thinking remains disabled. Native conditions initialize a spatially uniform
front at the configured budget. Routed
conditions prefill through the penultimate prompt token with that initial front and collect routing
signals across all selected Full-Attention heads/layers. Split–Merge then updates the front, and the
final prompt token runs as one cached token through every routed layer with the updated front. Its
logits produce the first generated token. This is one prefix prefill plus one single-token forward,
not a second prompt pass. Decode continues from the updated front, with further per-step updates only
in dynamic conditions. The full prompt cache is retained for bookkeeping, so this controls visual
information access but is not physical prompt-cache compression. First-token agreement with `full`
is reported as a diagnostic rather than enforced as an invariant.

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

The shared processed high-resolution grid is aligned to eight and capped at 4096 LLM visual tokens
per image. The configured `max_pixels` values are `4096*28^2` for Qwen2.5-VL and `4096*32^2` for
Qwen3.5. Native representation banks remain `/2`, `/4`, and `/8` in spatial resolution; the
configurable value controls the common active-token budget rather than those local bank scales.

With the current prepared manifest, A and C each contain 16,074 samples per condition. B contains
400 samples per condition. Therefore C evaluates `16,074 × 4 = 64,296` sample-condition pairs per compression
ratio and `321,480` across the default five-ratio sweep, per model. C reuses `formal/full` from A as
its high-resolution baseline, so A must be complete before standalone C analysis.

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
  --condition dynamic8_top8_native --task vstar_bench

# Run only ratio 6 (for C this narrows the configured sweep to one ratio).
python -m experiments.e4 run --model qwen25 --suite formal --compression-ratio 6
python -m experiments.e4 run --model qwen25 --suite compression --compression-ratio 6
python -m experiments.e4 analyze --model qwen25 --suite compression --compression-ratio 6
python -m experiments.e4 report --model qwen25 --suite compression --compression-ratio 6

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
