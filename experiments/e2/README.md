# E2 coarse visual representation experiment

E2 compares high-resolution visual-token pooling against native low-resolution inputs. It is independent of
`FoveaConfig`, E1 Head selection, and attention routing. The complete fine-grained cache is intentionally retained;
this experiment tests representation quality, not physical KV-cache compression.

## Conditions

- Full and native low-resolution `/2`, `/4`, and random-budget baselines.
- Uniform 2×2 and 4×4 pooling with KV-Center, Hidden-Center, and PostRoPE-KV.
- Native Uniform area-scale 4 and 16 conditions, using native `/2` and `/4` layer K/V with
  high-resolution region-center RoPE coordinates.
- Fixed and per-decode-step random multiscale fronts with 50%/30%/20% image area assigned to 1×1/2×2/4×4.
- Native fixed and per-step random conditions using the same fronts as their pooled counterparts.

Fixed pooled conditions apply the same coarse representation to prompt text prefill and decode. Per-step random
conditions use full prefill and resample a legal local front for every decode step.

There are 20 conditions. E2 evaluates native scales 1/4/16; the shared implementation also supports
area scale 64 (8×8 nodes), but E2 does not schedule that scale. Native conditions perform two auxiliary
prefills (`/2` and `/4`), retain only full-attention-layer visual K/V, and report their extra prefill time.

## NVIDIA environment

```bash
git submodule update --init
pip install -e third_party/lmms-eval
pip install --upgrade "huggingface-hub<1" transformers accelerate datasets pillow pyyaml qwen-vl-utils
export PYTHONPATH="$PWD/src:$PWD"
```

## Run

```bash
python -m experiments.e2 prepare --config experiments/e2/configs/default.yaml
python -m experiments.e2 run --model qwen25
python -m experiments.e2 run --model qwen35
python -m experiments.e2 analyze --model qwen25
python -m experiments.e2 report --model qwen25
```

Run or resume one condition with:

```bash
python -m experiments.e2 run --model qwen25 --condition uniform2_kv_center
python -m experiments.e2 run --model qwen25 --condition native_uniform4
python -m experiments.e2 run --model qwen25 --condition native_uniform16
python -m experiments.e2 run --model qwen25 --condition random_fixed_native
python -m experiments.e2 run --model qwen25 --condition random_perstep_native
```

Outputs are stored under `outputs/e2/<model>/`. Completed condition result files are reused on restart.
