# Scaling a 7B Vision-Language Model from One GPU to Two with PyTorch FSDP

**A profiling-driven study of the communication bottleneck in ZeRO-3 sharded training.**

> Full-parameter fine-tuning of Qwen2.5-VL-7B whose ~112 GB of training state *cannot* fit
> on a single 96 GB GPU. I shard it across two GPUs with PyTorch FSDP `FULL_SHARD` (ZeRO-3),
> profile the result with `torch.profiler`, find it is communication-bound on a no-NVLink
> PCIe link, and raise throughput **4.2×** — then prove with a fixed-seed control experiment that the speedup costs nothing in > training quality.

| | |
|---|---|
| **Stack** | PyTorch 2.9 (`+cu128`), FSDP1 `FullyShardedDataParallel`, `torch.profiler`, AdamW, activation checkpointing |
| **Hardware** | 2× NVIDIA RTX PRO 6000 Blackwell, 96 GB each, no NVLink, two NUMA nodes (GPU↔GPU over PCIe) |
| **Model** | Qwen2.5-VL-7B-Instruct, full-parameter SFT, frozen vision tower |
| **Headline** | OOM on 1 GPU → 75 GB/GPU on 2 · 690 → 2903 tok/s (4.2×) · MFU 3.1% → 13.1% · exposed comms 70% → 18% |

---

## TL;DR

- A 7B full fine-tune in the standard mixed-precision layout needs 16 bytes/parameter of
  optimizer state — ~112 GB, which overflows a single 96 GB card. I reproduced the exact
  failure: the run fills the card to 94.8 GB and then dies *inside the optimizer step* allocating
  Adam's first moment. This is the concrete reason sharding is *required*, not optional.
- FSDP `FULL_SHARD` (ZeRO-3) across 2 GPUs shards parameters, gradients, and optimizer
  state, bringing peak memory to ~75 GB/GPU — the model that OOM'd now trains with headroom.
- Profiling shows the run is communication-bound: at the naïve configuration over 70% of
  step time is exposed communication — the GPU stalling on all-gather / reduce-scatter over a
  no-NVLink, cross-NUMA PCIe link.
- I closed most of that gap with two well-understood levers, raising throughput **4.2×
  (690 → 2903 tok/s, MFU 3.1% → 13.1%)** and collapsing exposed comms from **70% to 18%**:
  - **Larger micro-batches** amortize the *fixed* per-micro-batch collectives over more tokens
    *and* grow the compute kernel that hides the all-gather behind it.
  - **`no_sync` gradient accumulation** drops reduce-scatter rounds **8 → 1**, worth **+63%** in
    the memory-bound regime where micro-batch size is capped.
- A fixed-seed `micro_bsz=1` vs `micro_bsz=8` control confirms the optimization trajectory is
  identical to floating-point precision (**max 0.53% loss drift over 30 steps**). The 4.2× is a
  systems win with zero quality cost — `micro_bsz` is a throughput knob, not a quality knob.

```
Throughput vs micro-batch size   (effective batch fixed at 16, no_sync off)

  mb=1  ██████                       690 tok/s   (3.1% MFU)
  mb=2  ███████████                 1275 tok/s   (5.8% MFU)
  mb=4  █████████████████           2040 tok/s   (9.2% MFU)
  mb=8  █████████████████████████   2903 tok/s   (13.1% MFU)  ← 4.2× baseline

Exposed communication % (GPU stalled on PCIe collectives) — collapses as compute hides it

  mb=1  ██████████████████  70%
  mb=2  ██████████████      56%
  mb=4  ███████████         43%
  mb=8  ████                18%   ← approaching compute-bound
```

---

**Scope** The goal here is distributed-training mechanics, not a task-accuracy result. I picked a different, larger model than the research champion precisely because 7B is the size that stops fitting on one card — so there is **no accuracy comparison** in this report, and none is owed: fairness matters for algorithm claims, and this is a systems claim. Where I made
a design choice, I state the alternative I rejected and why.

---

## 1. Why one GPU isn't enough

Full-parameter training with AdamW in the standard mixed-precision layout costs a fixed
16 bytes per parameter, independent of batch size:

| component | dtype | bytes/param |
|---|---|---|
| weights (compute copy) | bf16 | 2 |
| gradients | bf16 | 2 |
| master weights | fp32 | 4 |
| Adam `exp_avg` (first moment) | fp32 | 4 |
| Adam `exp_avg_sq` (second moment) | fp32 | 4 |
| total| | 16 |

`7e9 params × 16 B ≈ 112 GB` of training state — before a single activation. That overflows a
96 GB card, and that overflow is the entire motivation for ZeRO-3 sharding.

the model is loaded in fp32, not bf16. FSDP's `MixedPrecision(param_dtype=bf16)` casts to bf16 for compute and communication but keeps the fp32 master shard, and AdamW then holds fp32 moments — exactly the 16 B/param recipe above. Loading in bf16 instead would give bf16 optimizer states (~56 GB) that would fit on one card — collapsing the premise (and yielding a less numerically stable full fine-tune). The vision tower is frozen (`requires_grad_(False)` on `visual.*`), so it carries no gradients or optimizer state

Running on a single GPU (where FSDP degrades to `NO_SHARD`), the fp32 master weights plus gradients fill the card to 94.82 GB, and the run then OOMs inside `optim.step()` — the crash site is `adam.py → _init_group → torch.zeros_like`, i.e. precisely the allocation of Adam's `exp_avg`. It can't find the last ~260 MB. 

---

## 2. FSDP `FULL_SHARD` (ZeRO-3)

### Design decisions

| decision | why | alternative rejected |
|---|---|---|
| SFT, not GRPO | Isolates the distributed-training lesson cleanly. | GRPO+FSDP stacks three hard things — FSDP training + distributed generation + syncing *sharded* weights back into the sampler. Out of scope (understood conceptually, not built). |
| 7B, not 3B | A 7B full fine-tune (~112 GB) genuinely overflows one card. | A 3B full fine-tune (~48 GB) fits on one 96 GB card, so sharding would be contrived theatre. |
| 2 GPUs, not 3–4 | `112 GB / 2 ≈ 56 GB/GPU` of persistent state fits a 96 GB card with activation checkpointing + frozen vision. | More GPUs only buy comfort; the first size that *needs* 3 is ~13B (~208 GB). |
| No-NVLink box | Makes comms a visible fraction of step time — which is what makes the profiling story real. | An NVLink box would hide the bottleneck and make `backward_prefetch` look free. |

### The wrap

I used the classic FSDP1 API (`FullyShardedDataParallel`) deliberately — it is the most
documented and the concepts (`FULL_SHARD`, auto-wrap, prefetch, mixed-precision policy) are
identical to FSDP2's `fully_shard`.

- **`FULL_SHARD` (ZeRO-3):** shard parameters **+** gradients **+** optimizer state across ranks.
- **`transformer_auto_wrap_policy` per decoder block** (`Qwen2_5_VLDecoderLayer`): only one block's
  full parameters are ever materialized at a time → bounds peak memory.
- **`MixedPrecision(param_dtype=bf16, reduce_dtype=fp32, buffer_dtype=bf16)`:** bf16 compute,
  **fp32 gradient reduction** for numerical stability.
- **`BackwardPrefetch.BACKWARD_PRE`:** prefetch the *next* block's all-gather *during* the current
  block's backward, so the all-gather is hidden behind compute. On a no-NVLink box this is the key lever.
- **`limit_all_gathers=True`, `use_orig_params=True`** (the latter keeps real parameter shapes so
  optimizer param-groups work normally).
- **Activation checkpointing** applied *after* the FSDP wrap, non-reentrant, per decoder block —
  recompute activations in the backward pass instead of storing them, trading ~30% extra compute
  for a large activation-memory saving.

Gradient clipping uses FSDP's own `model.clip_grad_norm_`, which all-reduces the norm across
shards (a plain `torch.nn.utils.clip_grad_norm_` would clip each rank's shard independently and
get the global norm wrong).

Result: sharding ~112 GB of state across 2 cards lands at peak ~75 GB/GPU — the model that
OOM'd on one card now trains with ~20 GB of headroom. ✅

---

## 3. The run is communication-bound

### How I measured it

I profiled with `torch.profiler` and wrote a small **chrome-trace analyzer** that classifies every
on-device kernel as **compute** or **communication** (NCCL collectives, by kernel-name tokens —
`allgather` / `reducescatter`), then uses **interval-union math** to split communication into:

- **hidden comms** — overlapped with compute (i.e. `backward_prefetch` is working), and
- **exposed comms** — comms during which the GPU has *no* compute to do, so it stalls.

**Exposed-comms % of the step is the headline diagnostic**: it is the fraction of wall-clock the
GPU spends waiting on the PCIe / cross-NUMA link with nothing else to do. The analyzer is
stdlib-only and ships with a `--selftest` that validates the interval math on a synthetic trace.

> **A note on profiling rigor.** Exposed-comms % is **window-dependent** — it changes with which
> micro-batches the profiler captures. Because `no_sync` defers its reduce-scatter to the *last*
> micro-batch of an accumulation window, a fair measurement requires the profiled window to span a
> whole accumulation cycle. All comms-breakdown numbers in this report are measured over such an
> **accumulation-aligned window** (the profiler schedule is CLI-configurable for exactly this
> reason), so they are internally comparable across every configuration below.

### MFU, and what "good" looks like on this card

I report **MFU at the useful 6N FLOPs/token** (2N forward + 4N backward). With activation
checkpointing the hardware actually does ~8N (the extra forward in the backward pass) — that's HFU;
I report the stricter 6N number on purpose ("how much of peak goes to *useful* work").

The card's *achievable* bf16 GEMM (fp32-accumulate, the training path) measures **~401 TFLOPs/GPU**,
about **80%** of the theoretical dense **503.8 TFLOPs** (NVIDIA's marketed "1 PFLOP BF16" is the
2:4-*sparse* number; dense is half). MFU here is reported against the 503.8 theoretical, so even a
*perfectly* comms-hidden loop tops out near **~80% MFU** on this hardware. That sets the scale: the
question isn't "why not 100%," it's "how much of the comms gap can I close."

---

## 4. Closing the gap — two levers, one bottleneck

**Every run below holds the effective batch fixed at 16** (`micro_bsz × grad_accum × world_size`),
so these are pure *systems* comparisons — same optimization math, different execution.

| config (`micro_bsz × grad_accum`) | `no_sync` | tok/s | MFU | peak mem | compute % | comms % | **exposed %** | overlap % |
|---|---|---|---|---|---|---|---|---|
| `1 × 8` (baseline) | off | 690 | 3.1% | 75.5 GB | 24.0% | 81.6% | **70.3%** | 13.9% |
| `1 × 8` | **on** | 1126 | 5.1% | 75.5 GB | 31.3% | 72.1% | **56.9%** | 21.2% |
| `4 × 2` | off | 2040 | 9.2% | 83.9 GB | 46.2% | 69.2% | **42.9%** | 38.0% |
| `8 × 1` | off | **2903** | **13.1%** | 83.8 GB | 63.0% | 54.1% | **18.4%** | 65.9% |
| `4 × 2` | on | 2383 | 10.8% | 83.9 GB | 50.8% | 62.3% | **34.1%** | 45.2% |

### Lever 1 — bigger micro-batches amortize *and* hide the collectives

The FSDP collectives are a roughly **fixed cost per micro-batch**: the same parameters get
all-gathered and reduce-scattered regardless of how many tokens flow through (the all-gather kernel
count and absolute time are nearly identical across micro-batch sizes — ~57 all-gathers/micro-batch
either way). So doubling tokens-per-micro-batch does two things at once:

1. **Amortizes** the fixed comms tax over 2× the useful work, and
2. **Grows the compute kernel** that `backward_prefetch` hides the next all-gather behind.

Both show up in the trace: as `micro_bsz` goes 1 → 2 → 4 → 8, **compute rises 24% → 33% → 46% →
63%**, **overlap rises 14% → 24% → 38% → 66%**, and **exposed comms collapses 70% → 56% → 43% →
18%**. By `micro_bsz=8` the run is **approaching compute-bound** — the no-NVLink comms is now mostly
*hidden*, not eliminated. And it's nearly free on memory: peak climbs only **75.5 → ~84 GB** across
the whole sweep, because activations stay cheap under checkpointing.

**Net: 690 → 2903 tok/s = 4.2×**, MFU **3.1% → 13.1%**, with mild diminishing returns per doubling
(1.85× → 1.60× → 1.42×) as the run runs out of exposed comms to recover.

### Lever 2 — `no_sync` removes redundant reduce-scatter rounds

By default FSDP fires its gradient **reduce-scatter inside every `backward()`**. Under gradient
accumulation that is wasteful: you only need the combined gradient **once per optimizer step**, not
`grad_accum` times. `no_sync()` is a context manager that says *"don't run the gradient collective
this backward — accumulate locally"*; you wrap every micro-batch **except the last**.

Apples-to-apples at the same `1×8` config and the same profiler window:

| `1 × 8` | tok/s | MFU | reduce-scatter kernels | RS time | exposed comms | peak mem |
|---|---|---|---|---|---|---|
| baseline (sync every backward) | 690 | 3.1% | 465 | 12.3 s | 70.3% | 75.5 GB |
| **`no_sync`** (defer to last micro-batch) | **1126** | 5.1% | **58 (8.0× fewer)** | **1.5 s (8.0×)** | 56.9% | 75.5 GB |

- **Reduce-scatter rounds drop exactly 8× — and so does RS time** (12.3 s → 1.5 s): precisely the
  `grad_accum=8` amortization (8 RS rounds/step → 1). **All-gather is untouched** (912 → 912):
  `no_sync` defers only the *gradient* collective, not the parameter all-gather.
- **+63% throughput** (690 → 1126), exposed comms 70.3% → 56.9%.
- **The textbook cost of `no_sync` did not bite here.** While accumulating without syncing, each
  rank holds the **full, unsharded gradient** instead of its 1/world shard — normally a memory hit.
  At bf16 that's ~7.5 GB extra, comfortably absorbed by the ~20 GB headroom; **peak memory was
  essentially unchanged (75.5 GB)**. Knowing *when* a documented trade-off actually materializes is
  half the skill.
- **The bottleneck then shifts:** once RS is cheap, **parameter all-gather becomes the dominant
  comms** — which is exactly what Lever 1 attacks.

### The insight that ties them together

**The two levers attack the *same* cost — reduce-scatter rounds — from opposite ends:**

- Raising `micro_bsz` at fixed effective batch *lowers* `grad_accum`, which both cuts RS rounds
  (8 → 4 → 2 → 1) **and** grows the compute that hides the all-gather.
- `no_sync` cuts RS rounds at *fixed* `grad_accum`.

So once you can afford **`micro_bsz=8` (`grad_accum=1`)**, you've already driven RS rounds to the
floor of 1 — and **`no_sync` adds nothing** (it's a no-op at `grad_accum=1`). This is why `mb8`
(2903 tok/s) beats `mb4_ns` (2383): paying for `no_sync` instead of just using a bigger micro-batch
leaves throughput on the table.

**`no_sync` is the lever for when memory caps `micro_bsz` below your target effective batch**,
forcing `grad_accum > 1` — e.g. wanting effective batch 64 on a card that only fits `micro_bsz=8`
→ `grad_accum=4` → `no_sync` recovers the 4× RS reduction you *can't* get by raising `micro_bsz`
(you're out of memory). For *this* setup (effective batch 16, 7B fits at `micro_bsz=8`), the
throughput-optimal answer is simply **`micro_bsz=8` / `grad_accum=1`**.

---

## 5. Did the speedup cost any training quality? (a control experiment)

Tuning `micro_bsz` for throughput at fixed effective batch is only legitimate if the optimization
trajectory is genuinely *invariant* to it. So I tested it directly, as a **correctness** check.

**Setup.** `equiv_mb1` (`1×8`) vs `equiv_mb8` (`8×1`), **identical `--seed 42`** → identical data
order (the `DistributedSampler` permutation is seed-determined and batch-size-independent, so both
see the *same* 16 examples per optimizer step, on the same ranks). Both use **token-weighted
accumulation** (sum-of-token-losses ÷ a fixed shared norm), which is associative across the
micro-batch partition → the per-step gradient is **`micro_bsz`-invariant by construction**. 30
steps, no profiler, comparing the step-token-weighted mean loss.

| step | `mb1` (1×8) | `mb8` (8×1) | rel diff |
|---|---|---|---|
| 2 | 0.61530 | 0.61610 | 0.13% |
| 10 | 0.43140 | 0.43080 | 0.14% |
| 20 | 0.41830 | 0.41880 | 0.12% |
| 30 | 0.29980 | 0.29850 | 0.43% |

**Max absolute diff 0.0022; max relative diff 0.53% over 30 steps** (typical step ~0.1–0.2%).

**Interpretation.** The residual is **pure floating-point non-associativity, not a bug.** With
attention dropout = 0 (verified in the config) the forward pass is deterministic, so the *only*
difference between `mb1` and `mb8` is arithmetic order: `mb1` runs `[1, seq, d]` GEMMs while `mb8`
runs `[8, seq, d]` GEMMs, so cuBLAS picks different kernels/tiling → different summation order →
tiny logit deltas (plus fp32 reduce-scatter order). These stay **bounded under ~0.5%** rather than
exploding — the signature of numerical noise. A real accumulation bug (a missing `1/grad_accum`, a
broken gradient sync) would produce an **~8× gradient** and order-of-magnitude divergence within a
step or two. So the check **validates the accumulation + reduce-scatter path** *and* establishes the
takeaway:

> At fixed effective batch with correct token-weighted accumulation, **`micro_bsz` is a systems
> knob, not a quality knob.** The 4.2× throughput gain is free of any training-quality cost, to
> floating-point precision. The quality lever is the *effective batch* (held fixed throughout), not
> how it is chunked into micro-batches.

---

## 6. Concepts in brief (for the non-specialist reader)

**ZeRO-3 / FSDP `FULL_SHARD`.** Data-parallel training normally replicates the full model,
gradients, and optimizer state on *every* GPU. ZeRO-3 instead **shards all three** across ranks:
each GPU permanently holds only its `1/N` slice. To run a layer, ranks **all-gather** the full
parameters just-in-time, compute, then discard the gathered copy; in the backward pass they
**reduce-scatter** gradients so each rank ends up with only its slice. This is what makes a model
that doesn't fit on one GPU fit across many — at the cost of the all-gather / reduce-scatter traffic
that this report profiles.

**Gradient accumulation.** Run several small **micro-batches** through forward+backward *without*
stepping the optimizer, letting `.grad` pile up; step once after `grad_accum` of them. Because
gradients are additive, `N` micro-batches of size `M` ≈ one batch of size `N·M` — a large
*effective* batch under a memory ceiling. **Effective batch = `micro_bsz × grad_accum × world_size`**
(held at 16 here).

**`no_sync`.** In data-parallel training the per-GPU gradients must be combined before the optimizer
step (**DDP: all-reduce; FSDP: reduce-scatter**). By default that collective fires inside *every*
backward — but with accumulation you only need it *once* per optimizer step. `no_sync()` skips it on
the non-final micro-batches and accumulates locally, so reduce-scatter rounds drop by `grad_accum`×.
The trade-off: while accumulating un-synced, FSDP can't scatter the gradients, so each rank
temporarily holds the **full unsharded gradient** → more memory.

---

## 7. What I'd do next

- **Push `micro_bsz` further** until compute fully saturates or the cross-NUMA floor dominates —
  activations stay cheap under checkpointing, so there's headroom on memory.
- **FSDP2 (`fully_shard`)** — the newer per-parameter-sharding API; same concepts, cleaner ergonomics.
- **`sync_module_states` + meta-device init** to avoid every rank holding a full CPU copy of the
  weights at load time (the fp32 7B currently loads slowly off a network volume).
- **GRPO + FSDP** — the harder, research-adjacent version: FSDP training **+** distributed
  generation **+** syncing *sharded* weights back into the sampler. The natural next artifact.
- **CPU offload** as a last resort for even larger models (slow over PCIe, but unlocks scale).

---

## 8. What this demonstrates

- **Memory accounting for mixed-precision training** — deriving the 16 B/param recipe, and knowing
  that loading in fp32 (not bf16) is what forces the fp32 master + fp32 Adam moments that overflow the card.
- **FSDP / ZeRO-3 internals** — sharding params/grads/optimizer, transformer auto-wrap, backward
  prefetch, the mixed-precision policy (bf16 compute, fp32 reduction), and FSDP-aware gradient clipping.
- **Performance profiling** — `torch.profiler`, a custom trace analyzer that separates *hidden* from
  *exposed* communication via interval-union math, and the discipline to align the profiling window
  to the thing being measured.
- **Systems reasoning** — diagnosing a communication bottleneck, understanding *why* two knobs both
  help (and why they're redundant past a point), and reading the bottleneck *shift* after a fix.
- **Correctness rigor** — a fixed-seed control that distinguishes floating-point noise from an
  accumulation bug, and the precision (token-weighted loss) needed to make the comparison meaningful.
- **Honest scoping** — building the *smallest* setup that genuinely requires the technique, and being
  explicit about what the artifact does and does not claim.

---

## 9. Reproduction

```bash
# CPU smoke test (FSDP refuses CPU, so this validates the loop via a DDP fallback):
torchrun --standalone --nproc_per_node 2 \
    scripts/PGPS9K/9_train_sft_fsdp.py --smoke --steps 20

# 1-GPU OOM demonstration (the "why shard" evidence — OOMs inside optim.step()):
python -m torch.distributed.run --standalone --nproc_per_node 1 \
    scripts/PGPS9K/9_train_sft_fsdp.py --steps 2 --no_save

# 2-GPU real run, profiled:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
python -m torch.distributed.run --standalone --nproc_per_node 2 \
    scripts/PGPS9K/9_train_sft_fsdp.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --output_dir outputs/PGPS9K/sft7b_fsdp \
    --micro_bsz 8 --grad_accum 1 --steps 20 --no_save --profile --peak_tflops 503.8
    # add --no_sync to defer reduce-scatter when grad_accum > 1

# Analyze the comms breakdown (compute vs hidden vs exposed) from the trace:
python scripts/PGPS9K/analyze_fsdp_trace.py outputs/PGPS9K/sft7b_fsdp/trace

# Measure the card's achievable bf16 TFLOPs (feeds --peak_tflops):
python scripts/PGPS9K/measure_peak_tflops.py

# Fixed-seed micro-batch-invariance check (Section 5), token-weighted accumulation:
python -m torch.distributed.run --standalone --nproc_per_node 2 \
    scripts/PGPS9K/9_train_sft_fsdp.py --micro_bsz 1 --grad_accum 8 \
    --token_weighted_loss --loss_norm 4096 --seed 42 --steps 30 --no_save
```

**Key files (in this repository):**

- `scripts/PGPS9K/9_train_sft_fsdp.py` — the FSDP training script (wrap, mixed precision, prefetch,
  activation checkpointing, `no_sync`, token-weighted accumulation, MFU reporting, checkpoint gather).
- `scripts/PGPS9K/analyze_fsdp_trace.py` — the chrome-trace analyzer (compute vs hidden/exposed comms;
  `--selftest` validates the interval math with no GPU).
- `scripts/PGPS9K/measure_peak_tflops.py` — the achievable-bf16-GEMM probe behind `--peak_tflops`.

---

*Hardware: 2× NVIDIA RTX PRO 6000 Blackwell (96 GB, no NVLink). Software: PyTorch 2.9.0+cu128.
All numbers are from hands-on runs, June 2026.*
