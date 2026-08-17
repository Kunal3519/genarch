# Reproducing the paper

Every number in the paper, the script that produces it, and what to expect.

Assumes the dataset is in place (`docs/DATASET.md`) and the released checkpoints
are under `$GENARCH_CHECKPOINTS` (default `./checkpoints`). All commands run from
the repo root on one GPU unless noted.

Released checkpoints used throughout:

| stage | checkpoint |
|---|---|
| tokenizer | `ibq_4x_nc1024_nq1` |
| masked transformer | `myarch_mask_img_patch_v8_2` / `net_best_iou.tar` |
| refinement | `pool_classifier/poolcls_c_sym_v1` / `best.tar` (tau 0.45) |

---

## Table 1: main comparison (`tab:main`)

GenArch's row, on the full 40,000-tower validation split:

```bash
python script/eval_pipeline.py --split val40k --gpu_id 0 \
    --shard_idx $i --num_shards 6            # shard across GPUs
python script/eval_pipeline.py --combine --num_shards 6
```

Expect **IoU 0.9950**, 85.1% of towers ≥ 0.99, CD 0.0050, F@0 0.9975, F@1 1.000.

The two baseline rows come from separate codebases and are not reproducible from
this repo: *ArchComplete-AR* (0.9865) is an ArchComplete-style 3D-VQ
autoregressive transformer, *BrickGPT* (0.7801) a LoRA fine-tune of
Llama-3.2-1B emitting coordinate tokens. Both were retrained on this dataset
with the same four-view DINOv2 conditioning; see the paper for their settings.

**Runtime note.** A 6-shard full run takes a few hours. For a smoke check use
`--limit 200`, which lands near **single 0.9735 / refined 0.9916**, slightly
under the headline, as expected for a small slice.

## Table 2: inference stages (`tab:refine`)

Same run as above; `eval_pipeline.py` prints all three columns together.

| | single | vote (N=16) | + refinement |
|---|---:|---:|---:|
| IoU | 0.9802 | 0.9887 | **0.9950** |

## Table 3: number of conditioning views (`tab:views`)

```bash
python script/ablations/ablate_views.py --n_towers 1000 --vote --gpu_id 0
```

Mean over all view subsets of each size, deployed pipeline (N=16 + refinement):

| views | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| IoU | 0.892 | 0.961 | 0.986 | **0.995** |

Also reports the per-view marginals and the adjacent-vs-diagonal pair gap.

## Table 4: quantizer and objective (`tab:repr`)

Three trained models, not a sweep. Each row needs its own tokenizer and
transformer:

| quantizer | objective | IoU |
|---|---|---:|
| EMA residual VQ | CE only | 0.881 |
| EMA residual VQ | CE + coord loss | 0.901 |
| **IBQ** | **CE + coord loss** | **0.977** |

```bash
# row 3 (the released model)
torchrun --nproc_per_node=N script/train_vq.py --name <vq> --quantizer ibq \
    --down_t 2 --stride_t 2 --window_size 64 --nb_code 1024
torchrun --nproc_per_node=N script/train_transformer.py --name <run> \
    --vq_name <vq> --use_cls_prefix --coord_loss_weight 1.0 --ce_loss_weight 0.1

# rows 1-2: --quantizer ema --num_quantizers 3, then
#           --coord_loss_weight 0.0 (row 1) or 1.0 (row 2)
```

Backbone, conditioning, codebook size and token sequence are identical across
rows; only the quantizer and objective change.

## Table 5: decoding iterations and schedule (`tab:iters`)

```bash
python script/ablations/sweep_sampling_knobs.py --sweep timesteps --gpu_id 0
python script/ablations/sweep_sampling_knobs.py --sweep schedule  --gpu_id 0
```

400 validation towers. Refined IoU peaks at **T=10 (0.9958)** and *declines* by
T=30 even though single-pass keeps improving. Pick decoding knobs at the
deployed stage, not the single-pass one.

| | T=8 | T=10 | T=15 | T=20 | T=30 |
|---|---:|---:|---:|---:|---:|
| single | 0.9797 | 0.9815 | 0.9844 | 0.9859 | **0.9864** |
| refined | 0.9952 | **0.9958** | 0.9954 | 0.9955 | 0.9945 |
| ms/tower | 137 | 163 | 237 | 312 | 460 |

Cosine vs linear schedule: 0.9958 vs 0.9949 refined.

## Table 6: nucleus and top-k truncation (`tab:sampling`)

```bash
python script/ablations/sweep_sampling_knobs.py --sweep topk_topp --gpu_id 0
```

All settings land within ~0.0004 refined (0.9955–0.9958). The point of the table
is that these knobs are inert; the defaults are fine.

## Table 7: training masking ratio (`tab:maskratio`)

Five matched 100-epoch training runs, evaluated on 1,000 validation towers:

```bash
# ours
torchrun --nproc_per_node=1 script/train_transformer.py --name mr_cosine \
    --train_mask_dist cosine --max_epoch 100 --stream_img_feat ...
# arms
--train_mask_dist uniform --train_mask_alpha {0.2,0.4,0.6,0.8}
```

| training ratio | single | vote (N=16) | refined |
|---|---:|---:|---:|
| **cos(πu/2)** (ours) | **0.9590** | **0.9800** | **0.9888** |
| U(0.2,1) | 0.9459 | 0.9748 | 0.9851 |
| U(0.4,1) | 0.9497 | 0.9759 | 0.9860 |
| U(0.6,1) | 0.9572 | 0.9788 | 0.9886 |
| U(0.8,1) | 0.9551 | 0.9764 | 0.9865 |

Absolute values sit below the headline because this is a matched **100-epoch**
budget, not the converged run; the comparison is the relative ordering, which
holds at all three stages.

**Trap:** token accuracy is *not* comparable across these arms, because each grades a
different task difficulty, and less masking looks better for free. Only
downstream voxel IoU is a valid comparison.

## Inference speed (AITS)

```bash
python script/ablations/time_inference.py --mode e2e     --gpu_id 0
python script/ablations/time_inference.py --mode voting  --gpu_id 0
```

0.23 s/tower end to end on one H200 at batch one: ~46 ms single pass, ~177 ms for
the 16 drafts, ~52 ms pool + refinement.

## Qualitative figures

```bash
python script/gen_paper_diagrams.py  # teaser + architecture panels
python script/gen_compare.py         # GenArch reconstructions vs ground truth
```

The three-way comparison figure additionally needs each baseline's saved
coordinates, which ship with neither codebase.

---

## Things that will silently cost you accuracy

- **Decode IBQ whole-sequence.** Windowed decode (`w=64`) is correct for the
  residual-VQ tokenizer and wrong for IBQ, where it costs ~0.03 IoU. The reverse is
  worse: whole-sequence decode of residual VQ collapses to ~0.65.
- **Keep `cond_scale = 1.0`.** Classifier-free guidance was swept and is
  monotonically harmful here; voting degrades about twice as fast as single-pass
  under it.
- **The vote threshold is 0.4375** (≥7 of 16), not 0.3. The optimum moved up as
  the sample pool got cleaner.
- **`generate()` needs a token count.** All numbers above take it from ground
  truth. For image-only inference use the length predictor
  (`script/ablations/train_length_predictor.py`, MAE ~6.6 tokens) with symmetric
  length jitter across the 16 samples; expect ~0.980 rather than ~0.986.
