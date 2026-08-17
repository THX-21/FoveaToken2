# E3: Text-Anchor position encoding

E3 evaluates decode-only `text_anchor` on four fixed visual representations. Prompt prefill uses
the paired control representation and normal position encoding, so every paired condition must
produce the same first generated token.

Both models run with thinking disabled. Every condition receives the same structured instruction to
produce a natural `Analyze:` section of at most 200 words followed by a short `Answer:`. Only the
parsed answer is passed to the original lmms-eval scorer. The default generation cap is 320 tokens.

## Conditions

- Full resolution: `full_mrope` / `full_text_anchor`
- Native `/2` resolution: `lowres2_mrope` / `lowres2_text_anchor`
- High-resolution 2×2 raw-KV pooling: `pool2_center` / `pool2_text_anchor`
- Native `/2` K/V mapped to 2×2 nodes: `native2_center` / `native2_text_anchor`

E3 reuses E2's fixed 400-sample manifest, but all eight conditions are rerun because the structured
prompt differs from E2. This keeps each control/Text-Anchor pair directly comparable.

## Commands

```bash
python -m experiments.e3 prepare --config experiments/e3/configs/default.yaml
python -m experiments.e3 run --model qwen25
python -m experiments.e3 run --model qwen35
python -m experiments.e3 run --model qwen25 --condition pool2_text_anchor
python -m experiments.e3 analyze --model qwen25
python -m experiments.e3 report --model qwen25
```

To split the samples within each condition across four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  -m experiments.e3 run --model qwen35

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  -m experiments.e3 run --model qwen35 --condition full_text_anchor
```

Conditions are not assigned to different GPUs. Each condition is processed by all workers, with
its sample list sharded across ranks, then rank 0 merges and scores the complete result. Generation
is resumable. `reevaluate` is CPU/API concurrency and should continue to use plain `python` with
its `--workers` option.

Outputs are written to `outputs/e3/<model>/`. The full matrix contains 6,400 sample generations
across both models. Native2 captures only the `/2` auxiliary visual K/V bank.
