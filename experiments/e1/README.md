# E1 visual routing Head discovery

E1 discovers useful visual routing Heads without ground-truth ROI annotations. It uses natural-image
generation to measure visual mass, spatial concentration, and visual-only HybridKV dynamics, then
uses controlled 3x3 image grids to measure whether attention follows the requested panel.

The experiment is intentionally separate from `FoveaConfig`; it produces `head_selection.json`
files consumed by normal TokenFovea inference.

## NVIDIA environment

Use a fresh Python 3.11 environment. Qwen3.5 requires a recent Transformers build. A GPU with at
least 32 GB available memory is recommended for each worker.

```bash
git submodule update --init --recursive
python -m venv .venv-e1
source .venv-e1/bin/activate
pip install --upgrade pip
pip install -e third_party/lmms-eval
pip install --upgrade transformers "huggingface-hub<2" accelerate datasets pillow pyyaml
export PYTHONPATH="$PWD/src:$PWD"
```

Authenticate with Hugging Face if the Lite datasets or model repositories require it:

```bash
hf auth login
```

## Run

```bash
python -m experiments.e1 prepare --config experiments/e1/configs/default.yaml
python -m experiments.e1 probe --model qwen25
python -m experiments.e1 analyze --model qwen25
python -m experiments.e1 probe --model qwen25 --pass visualize
python -m experiments.e1 report --model qwen25
```

The single-command equivalent, which avoids reloading the model between the two probe passes, is:

```bash
python -m experiments.e1 run --model qwen25
python -m experiments.e1 run --model qwen35
```

To split samples across four GPUs, launch one model replica per GPU with `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  -m experiments.e1 run --model qwen35
```

This is sample data parallelism: every GPU processes a disjoint subset of the natural, controlled,
and visualization samples. Rank-local checkpoints under `outputs/e1/<model>/.distributed/` are
merged by rank 0, so interrupted runs remain resumable. Do not launch `analyze` or `report` with
`torchrun`; the `run` command already executes those rank-0-only stages.

The scan uses BF16, SDPA, batch size one, greedy decoding, and seed 42. It processes 100 images
from each of COCO Caption Lite, Flickr30k Lite, and NoCaps Lite, plus 100 controlled collages.
Natural generation is capped at 32 tokens. Qwen3.5 linear-attention layers are excluded.

## Outputs

Each `outputs/e1/<model>/` directory contains:

- `natural_metrics.json`, `gaze_metrics.json`, and `probe_metadata.json`;
- `head_metrics.csv/json` and `hybridkv_classification.json`;
- `head_selection_top4.json`, `head_selection_top8.json`, and `head_selection_top16.json`;
- `head_selection.json`, an alias of the default Top-8 selection;
- `visualization/attention_traces.jsonl` and `report.html` with its image assets.

The HybridKV label is descriptive only: both `stable_localizer` and `dynamic_gaze` Heads remain
eligible. Final ranking uses positive null-calibrated GazeScore after the visual-mass × concentration
filter.
