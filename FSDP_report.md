# Scaling a 7B Vision-Language Model From One GPU to Two With PyTorch FSDP
| | |
|---|---|
| **Stack** | PyTorch 2.9, FSDP1 (`FullyShardedDataParallel`), `torch.profiler`, AdamW, activation checkpointing |
| **Hardware** | 2× NVIDIA RTX PRO 6000 Blackwell, 96 GB each, no NVLink between them, connected over PCIe |
| **Model** | Qwen2.5-VL-7B-Instruct, full-parameter fine-tuning, vision encoder frozen |
| **Headline** | Crashes with out-of-memory on 1 GPU → runs at 75 GB/GPU on 2 · throughput 690 → 2903 tokens/sec (4.2×) · GPU utilization (MFU) 3.1% → 13.1% · time wasted waiting on communication 70% → 18% |

---

## TL;DR

A full fine-tune of a 7B-parameter model needs about 16 bytes of memory per parameter to
hold the optimizer's state — around 112 GB in total, which doesn't fit on a single 96 GB
card. I reproduced that failure directly: the run fills the card to 94.8 GB and then crashes
*while allocating memory inside the optimizer step*. That crash is the concrete reason
sharding the model across GPUs is required here, not just a nice-to-have.

Splitting the model with FSDP's `FULL_SHARD` mode (ZeRO-3) divides the parameters,
gradients, and optimizer state across both GPUs, bringing peak memory down to about
75 GB/GPU. The model that couldn't fit on one card now trains with room to spare.

Profiling that two-GPU run showed it was **communication-bound**: in the default
configuration, the GPUs spent over 70% of each training step just waiting for data to move
across the PCIe link between them, with no useful math happening in the meantime. I closed
most of that gap with two well-understood techniques, raising throughput **4.2× (690 → 2903
tokens/sec)** and cutting wasted communication time from **70% down to 18%**:

- **Larger micro-batches.** Each round of GPU-to-GPU communication has a roughly fixed cost,
  regardless of how many tokens it's carrying. Processing more tokens per micro-batch spreads
  that fixed cost over more useful work, and also gives the GPU more computation to run
  *during* each communication round — hiding it instead of waiting on it.
- **`no_sync` gradient accumulation.** This technique cuts the number of gradient-syncing
  communication rounds from 8 down to 1 per optimizer step, worth a **+63%** throughput gain
  on its own, in the regime where memory limits how large a micro-batch can be.

Finally, a controlled comparison — same random seed, `micro_bsz=1` vs. `micro_bsz=8` — shows
the model learns identically either way, with results matching to within normal
floating-point noise (**max 0.53% loss drift over 30 steps**). In other words, the 4.2×
speedup is a pure systems win: `micro_bsz` is a throughput knob, not a knob that affects
training quality.

```
Throughput vs micro-batch size   (effective batch fixed at 16, no_sync off)

  mb=1  ██████                       690 tok/s   (3.1% MFU)
  mb=2  ███████████                 1275 tok/s   (5.8% MFU)
  mb=4  █████████████████           2040 tok/s   (9.2% MFU)
  mb=8  █████████████████████████   2903 tok/s   (13.1% MFU)  ← 4.2× baseline

Time wasted waiting on GPU-to-GPU communication — shrinks as more work hides it

  mb=1  ██████████████████  70%
  mb=2  ██████████████      56%
  mb=4  ███████████         43%
  mb=8  ████                18%   ← approaching compute-bound
```

---

**A note on scope.** This report is about distributed-training *mechanics* — memory, sharding,
communication overlap — not about how accurate the fine-tuned model is on some task. I
deliberately chose a larger model than I'd use for an accuracy-focused project, because 7B is
the size where full fine-tuning stops fitting on a single card, which is exactly the problem
this report is about. So there's no accuracy benchmark here, and none is needed: that kind of
comparison matters when you're making a claim about an algorithm, and this is a claim about
systems performance. Wherever I made a design choice, I've noted the alternative I
considered and why I didn't go with it.

---

## 1. Why One GPU Isn't Enough

Full-parameter fine-tuning with the AdamW optimizer, in the standard mixed-precision setup,
requires keeping several copies of every parameter around simultaneously, each in a
different numeric format:

| what's stored | format | bytes per parameter |
|---|---|---|
| weights used for the actual math | bf16 (16-bit) | 2 |
| gradients | bf16 | 2 |
| "master" weights — the full-precision copy actually updated each step | fp32 (32-bit) | 4 |
| Adam's first moment (a running average used to smooth updates) | fp32 | 4 |
| Adam's second moment (a running average used to scale updates) | fp32 | 4 |
| **total** | | **16** |

For a 7-billion-parameter model, that's `7e9 × 16 bytes ≈ 112 GB` — and that's before
accounting for any of the memory activations need during the forward and backward pass. That
already overflows a 96 GB card, and that overflow is the entire reason sharding across GPUs
is necessary.

It's worth explaining *why* the numbers work out this way rather than assuming bf16 alone
would be enough. FSDP's mixed-precision setting casts the model to bf16 only for compute and
for communication between GPUs — it still keeps a full fp32 "master" copy of the weights,
and AdamW's internal moment tracking also stays in fp32. This is the standard, numerically
stable recipe, and it's what produces the 16-bytes-per-parameter total above. Loading the
whole model in bf16 from the start would shrink the optimizer state to roughly 56 GB, small
enough to fit on one card — but that would remove the very problem this report is about, and
full fine-tuning purely in bf16 is known to be less numerically stable. So this constraint is
deliberate, not an oversight. One more detail: the vision encoder is frozen (its weights
never get updated during training), so it needs no gradients or optimizer state at all — only
the language-model parameters carry the full 16-byte-per-parameter cost.

To confirm this isn't just arithmetic on paper, I ran the fine-tune on a single GPU (where
FSDP has nothing to shard, so it behaves like ordinary unsharded training). The fp32 master
weights and gradients alone filled the card to 94.82 GB out of 96 GB available. The run then
crashed inside the optimizer step, at the exact line of code that allocates Adam's
first-moment buffer — it needed roughly 260 MB more than was left on the card.

---

## 2. Sharding the Model With FSDP's `FULL_SHARD` Mode (ZeRO-3)

FSDP (Fully Sharded Data Parallel) is PyTorch's built-in tool for splitting a model's
parameters, gradients, and optimizer state across multiple GPUs, so no single GPU has to hold
all of it at once. Its `FULL_SHARD` mode implements what's commonly called **ZeRO-3** (stage
3 of the "Zero Redundancy Optimizer" approach): parameters, gradients, *and* optimizer state
are all split across GPUs, and each GPU only reconstructs the full parameters for one small
part of the model at a time — just long enough to use them, then releases them again.

### Design decisions

| decision | why | alternative considered and rejected |
|---|---|---|
| Supervised fine-tuning, not reinforcement learning (GRPO) | Keeps the focus purely on the distributed-training problem. | Combining GRPO with FSDP stacks three hard problems at once — FSDP training, distributed text generation, and keeping sharded weights in sync with the generation step. Out of scope for this report (I understand the approach conceptually, but didn't build it here). |
| 7B parameters, not 3B | A 7B full fine-tune (~112 GB) genuinely doesn't fit on one card. | A 3B fine-tune (~48 GB) fits comfortably on a single 96 GB card, so sharding it would be solving a problem that doesn't exist. |
| 2 GPUs, not 3–4 | 112 GB split two ways is ~56 GB/GPU of persistent state, which fits a 96 GB card once you add activation checkpointing and a frozen vision encoder. | More GPUs would only add comfort margin — the model size that actually *requires* 3 GPUs is closer to 13B parameters (~208 GB). |
| A machine without NVLink | Makes GPU-to-GPU communication a real, visible fraction of each step — which is exactly what makes the profiling results in this report meaningful. | A machine with NVLink would hide the bottleneck almost entirely and make some of the optimizations below look unnecessary. |

### How the model is wrapped

I used the original FSDP1 API (`FullyShardedDataParallel`) rather than the newer FSDP2,
deliberately — it's the more thoroughly documented option, and the core concepts
(`FULL_SHARD`, auto-wrapping, prefetching, mixed-precision policy) carry over directly to
FSDP2's `fully_shard` API.

- **`FULL_SHARD` (ZeRO-3):** parameters, gradients, and optimizer state are all sharded
  across the two GPUs.
- **Wrapping policy, one decoder block at a time** (`transformer_auto_wrap_policy` on
  `Qwen2_5_VLDecoderLayer`): only one transformer block's full parameters are ever
  reconstructed on a GPU at once, which bounds peak memory usage.
- **Mixed precision** (`param_dtype=bf16, reduce_dtype=fp32, buffer_dtype=bf16`): computation
  happens in bf16 for speed, but gradients are combined across GPUs in fp32 for numerical
  stability.
- **Backward prefetching** (`BackwardPrefetch.BACKWARD_PRE`): while one decoder block is still
  running its backward pass, FSDP starts pulling in ("all-gathering") the *next* block's
  parameters in the background. That overlaps communication with computation instead of doing
  them one after another. On a machine without NVLink, this single setting is the most
  important lever available before even touching batch size.
- **`limit_all_gathers=True`** and **`use_orig_params=True`** (the latter keeps parameters in
  their normal shapes, so the optimizer's parameter groups behave the way they would in
  unsharded training).
- **Activation checkpointing**, applied after the FSDP wrap, per decoder block: instead of
  storing activations from the forward pass for later use in the backward pass, they're
  recomputed on the fly. This trades about 30% more compute for a large reduction in memory
  used by activations.

Gradient clipping uses FSDP's own `model.clip_grad_norm_` rather than the standard PyTorch
version. This matters because a plain `torch.nn.utils.clip_grad_norm_` would only see each
GPU's *shard* of the gradients and compute the wrong (too-small) global norm; FSDP's version
correctly combines the norm across all shards first.

**Result:** splitting ~112 GB of training state across 2 GPUs brings peak memory down to
about 75 GB/GPU — the model that crashed on one card now trains with roughly 20 GB of
headroom to spare.

---

## 3. Diagnosing the Bottleneck: The Run Is Communication-Bound

### How I measured it

I profiled training with `torch.profiler` and wrote a small trace analyzer that classifies
every operation running on the GPU as either "compute" or "communication" (identifying NCCL
collective operations — GPU-to-GPU data transfers — by name, e.g. all-gather and
reduce-scatter). It then measures, for every point in time, whether communication is:

- **hidden** — happening while the GPU is also doing useful compute (meaning the prefetching
  described above is working), or
- **exposed** — happening while the GPU has nothing else to do, so it's simply stalled,
  waiting.

The percentage of a training step spent in exposed communication is the key diagnostic in
this report: it's the fraction of wall-clock time the GPU spends idle, waiting on data to
arrive over the PCIe link. (The analyzer itself uses only Python's standard library and ships
with a self-test that checks its timing math against a synthetic trace.)

> **A note on measurement rigor.** The exposed-communication percentage depends on exactly
> which micro-batches the profiler happens to capture. Because `no_sync` (explained in
> Section 4) defers its communication to the *last* micro-batch of an accumulation cycle, a
> fair measurement has to span a *complete* accumulation cycle, not an arbitrary slice of one.
> Every comparison in this report is measured over such a complete cycle, so the numbers are
> directly comparable across configurations.

### What "good" looks like on this hardware (MFU)

MFU (Model FLOPs Utilization) measures what fraction of the GPU's peak math throughput is
actually going toward useful work. I calculate it using the standard "6N FLOPs per token"
estimate (2N for the forward pass, 4N for the backward pass, where N is the parameter count).
Note that activation checkpointing means the hardware is actually doing more work than this —
closer to 8N, because part of the forward pass is recomputed during the backward pass — but I
report the stricter 6N number deliberately, since it reflects genuinely useful work rather
than recomputation overhead.

It also matters what 100% MFU would even mean on this specific hardware. This card's real,
measured bf16 throughput (with fp32 accumulation, matching the training setup) is about 401
TFLOPs/GPU — roughly 80% of its theoretical dense peak of 503.8 TFLOPs. (NVIDIA's marketed
"1 PFLOP BF16" figure refers to a sparse-computation mode that doesn't apply here; the dense
figure is about half that.) Since MFU here is measured against the 503.8 TFLOPs theoretical
peak, even a perfectly optimized run with zero exposed communication would top out around
80% MFU. That sets the right expectation: the interesting question isn't "why isn't this at
100%," but "how much of the communication-related gap can actually be closed."

---

## 4. Closing the Gap

Every configuration below holds the *effective* batch size fixed at 16
(`micro_bsz × grad_accum × world_size`), so these are pure systems comparisons: the
optimization math is identical in every row, and only the execution strategy changes.

| config (`micro_bsz × grad_accum`) | `no_sync` | tokens/sec | MFU | peak memory | compute % | comms % | **exposed %** | overlap % |
|---|---|---|---|---|---|---|---|---|
| `1 × 8` (baseline) | off | 690 | 3.1% | 75.5 GB | 24.0% | 81.6% | **70.3%** | 13.9% |
| `1 × 8` | on | 1126 | 5.1% | 75.5 GB | 31.3% | 72.1% | **56.9%** | 21.2% |
| `4 × 2` | off | 2040 | 9.2% | 83.9 GB | 46.2% | 69.2% | **42.9%** | 38.0% |
| `8 × 1` | off | 2903 | 13.1% | 83.8 GB | 63.0% | 54.1% | **18.4%** | 65.9% |
| `4 × 2` | on | 2383 | 10.8% | 83.9 GB | 50.8% | 62.3% | **34.1%** | 45.2% |

### Lever 1 — bigger micro-batches amortize the communication cost and hide it behind compute

FSDP's communication has a roughly fixed cost per micro-batch: the same parameters get
gathered and the same gradients get combined regardless of how many tokens are actually being
processed (the number and duration of all-gather operations is nearly identical across
micro-batch sizes — about 57 all-gathers per micro-batch either way). So increasing the
number of tokens per micro-batch does two things simultaneously:

1. **Amortizes** that fixed communication cost over twice the useful work, and
2. **Grows the compute step**, giving backward prefetching more work to hide the next
   all-gather behind.

Both effects show up clearly in the trace: as `micro_bsz` goes from 1 → 2 → 4 → 8, the
fraction of time spent on useful compute rises from 24% → 33% → 46% → 63%, the fraction of
communication that's successfully hidden behind compute rises from 14% → 24% → 38% → 66%, and
exposed (wasted) communication collapses from 70% → 56% → 43% → 18%. By `micro_bsz=8`, the run
is approaching compute-bound — the lack of NVLink is now mostly hidden rather than eliminated.
And this comes almost for free in memory terms: peak usage only climbs from 75.5 GB to about
84 GB across the whole sweep, since activation checkpointing keeps activations cheap.

**Net effect: 690 → 2903 tokens/sec, a 4.2× improvement, and MFU rising from 3.1% to 13.1%.**
The gains do show mild diminishing returns with each doubling (1.85× → 1.60× → 1.42×), which
makes sense: there's less exposed communication left to reclaim each time.

### Lever 2 — `no_sync` removes redundant communication rounds

By default, FSDP triggers a reduce-scatter (the operation that combines gradients across
GPUs and re-splits them into shards) on *every* backward pass. Under gradient accumulation,
that's wasteful — you only actually need the combined gradient once per optimizer step, not
once per micro-batch. `no_sync()` is a context manager that tells FSDP "skip the gradient
communication this backward pass, just accumulate locally instead"; you wrap every
micro-batch except the final one in it.

Comparing directly at the same `1×8` configuration:

| `1 × 8` | tokens/sec | MFU | reduce-scatter operations | time spent on them | exposed comms | peak memory |
|---|---|---|---|---|---|---|
| baseline (syncs every backward pass) | 690 | 3.1% | 465 | 12.3 s | 70.3% | 75.5 GB |
| **with `no_sync`** (syncs once, at the end) | **1126** | 5.1% | **58 (8× fewer)** | **1.5 s (8× less)** | 56.9% | 75.5 GB |

- The number of reduce-scatter operations — and the time spent on them — both drop by exactly
  8×, matching the `grad_accum=8` setting (8 sync rounds per step collapses to 1).
  All-gather activity is unaffected (912 either way): `no_sync` only defers the *gradient*
  communication, not the parameter-gathering step.
- This alone is worth a 63% throughput increase (690 → 1126 tokens/sec), and cuts exposed
  communication from 70.3% to 56.9%.
- While accumulating without syncing, each GPU temporarily holds the *full*, unsharded
  gradient rather than its normal 1/N shard — ordinarily a real memory cost. At bf16
  precision that's about 7.5 GB extra, which is comfortably absorbed by the ~20 GB of
  headroom freed up by sharding — peak memory stayed essentially flat at 75.5 GB. Knowing
  *when* a documented trade-off will actually bite you, versus when there's enough slack to
  absorb it, turned out to matter here.
- Once the gradient-sync cost is minimized, the parameter all-gather becomes the dominant
  source of communication time — which is exactly what Lever 1 addresses.

### How the two levers fit together

Both levers ultimately attack the same underlying cost — the number of reduce-scatter
rounds — just from opposite directions:

- Increasing `micro_bsz` at a fixed effective batch size *lowers* `grad_accum`, which both
  reduces the number of reduce-scatter rounds (8 → 4 → 2 → 1) *and* grows the compute that
  hides the remaining all-gather traffic.
- `no_sync` reduces reduce-scatter rounds directly, at a *fixed* `grad_accum`.

So once `micro_bsz=8` (`grad_accum=1`) is affordable, reduce-scatter rounds are already down
to the theoretical minimum of 1 per step — and `no_sync` has nothing left to contribute (it's
a no-op when `grad_accum=1`). That's why the best pure-`micro_bsz` configuration (`mb8`, 2903
tokens/sec) beats the best `no_sync`-assisted configuration (`mb4` with `no_sync`, 2383
tokens/sec): paying the extra bookkeeping cost of `no_sync` is worse than simply using a
larger micro-batch when memory allows it.

`no_sync` earns its keep specifically when memory limits `micro_bsz` below what your target
effective batch size would otherwise need — forcing `grad_accum > 1`. For example, wanting
an effective batch of 64 on a card that only fits `micro_bsz=8` means `grad_accum=4`, and
`no_sync` recovers the resulting 4× reduction in reduce-scatter rounds that you can't get any
other way (there's no memory left to raise `micro_bsz` further). For the setup used in this
report — effective batch 16, and the model fitting at `micro_bsz=8` — the straightforwardly
best answer is simply `micro_bsz=8`, `grad_accum=1`.

---

## 5. Does the Speedup Cost Any Training Quality? (A Control Experiment)

Tuning `micro_bsz` purely for throughput, while holding the effective batch size fixed, is
only a valid thing to do if the resulting optimization trajectory is genuinely unaffected by
it. So I tested that directly, as a correctness check rather than an assumption.

**Setup.** I compared `equiv_mb1` (`1×8`) against `equiv_mb8` (`8×1`) using an identical
`--seed 42` in both runs. Because the `DistributedSampler`'s shuffling is seed-determined and
independent of batch size, this guarantees both runs see the exact same 16 training examples
per optimizer step, in the same order, on the same GPUs. Both runs also use token-weighted
gradient accumulation (loss is summed across tokens, then divided by a fixed, shared
normalizer) — a method that's mathematically associative across however the micro-batches are
split, meaning the resulting per-step gradient is, by construction, invariant to `micro_bsz`.
I compared the two runs over 30 steps, using the token-weighted mean loss at each step (no
profiler running during this comparison).

| step | `mb1` (1×8) | `mb8` (8×1) | relative difference |
|---|---|---|---|
| 2 | 0.61530 | 0.61610 | 0.13% |
| 10 | 0.43140 | 0.43080 | 0.14% |
| 20 | 0.41830 | 0.41880 | 0.12% |
| 30 | 0.29980 | 0.29850 | 0.43% |

Maximum absolute difference: 0.0022. Maximum relative difference over the full 30 steps:
0.53% (a typical step differs by only 0.1–0.2%).

**What this means.** This small residual difference is ordinary floating-point noise, not a
bug. With attention dropout set to 0 (confirmed in the config), the forward pass is otherwise
deterministic, so the only thing that differs between `mb1` and `mb8` is the *order* in which
numbers get summed: `mb1` runs matrix multiplications shaped `[1, seq, d]` while `mb8` runs
`[8, seq, d]`, so the underlying GPU math library (cuBLAS) selects different kernels and
tiling strategies, which changes summation order and produces tiny differences in the
resulting values (plus a similar effect from the order of operations in the fp32
reduce-scatter). This kind of difference stays small and bounded — under 0.5% here — which is
the signature of harmless numerical noise. A genuine accumulation bug (for example, a missing
division by `grad_accum`, or a broken gradient sync) would instead produce a gradient roughly
8× too large and cause the two runs to diverge by an order of magnitude within just a step or
two — which is not what happened. So this check both validates that the accumulation and
reduce-scatter logic is correct, and supports the following conclusion:

> At a fixed effective batch size, with correctly implemented token-weighted accumulation,
> **`micro_bsz` is a systems-performance knob, not a training-quality knob.** The 4.2×
> throughput improvement comes at no cost to training quality, down to the level of ordinary
> floating-point precision. The setting that actually affects training quality is the
> *effective batch size* — which was held constant throughout every experiment in this
> report — not how that batch is split into micro-batches.

---

## 6. What I'd Do Next

- Push `micro_bsz` even higher until compute fully saturates the GPUs or the cross-GPU memory
  bottleneck takes over — there's still headroom, since activation checkpointing keeps
  activation memory cheap.
- Try FSDP2 (`fully_shard`), the newer API that shards at the level of individual parameters
  rather than whole modules.
- Add `sync_module_states` with meta-device initialization, so that only one GPU needs to load
  a full CPU copy of the weights at startup rather than every GPU doing so — the current setup
  loads the full fp32 7B model slowly from a network drive.
- Extend this setup to GRPO (reinforcement learning) combined with FSDP.
