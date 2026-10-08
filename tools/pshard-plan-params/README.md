# pshard-plan-params

`llama-pshard-plan-params` writes a pshard registry for a model and VRAM budget.

It probes batch-size tiers such as `bs=1`, `bs=16`, and `bs=512`, prices the allowed placement strategies at each tier from the machine profiles and the GGUF, and writes the fastest viable plan per tier to:

```text
<model>.gguf.tensor_overrides.pshard_registry
```

The planner needs this machine's CPU profile; it refuses to plan without one (see [Inputs](#inputs)). At runtime, pshard prices how to cut each batch across the tiers and switches plans between ubatches (see [Runtime tier choice](#runtime-tier-choice)).

The registry is plain text so it can be inspected when debugging planner output.

The tool always plans fresh: it does not reuse a stored plan. A tool started with `-pshard` that finds no plan for its configuration plans it in-process the same way, unless `--pshard-no-plan-on-miss` is set.

## Example usage

```bash
# 1. Profile the machine once, at the thread count the runs use
#    (writes cpu_profile.txt and gpu_profile.txt in the working directory)
./build/bin/llama-profiler-cpu --threads 8
./build/bin/llama-profiler-gpu

# 2. Plan placement for a model + VRAM budget (from the directory that holds the profiles)
./build/bin/llama-pshard-plan-params \
    --model /models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    --max-vram-alloc 8000 \
    -c 65536 -t 8

...
pshard_registry_save: saved budget=8000 MiB cache_ubatch=8192 variant with 6 tier plans to /models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf.tensor_overrides.pshard_registry
main: planning complete, registry written next to model file

# 3. Inspect the registry
cat /models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf.tensor_overrides.pshard_registry

# 4. Run with the same model, budget, context and threads, and one slot
#    (llama-server defaults to 4 slots with a unified KV cache, which the plan does not match)
./build/bin/llama-server -m /models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -pshard -mva 8000 -c 65536 -t 8 -np 1
```

Example registry excerpt (long lines cut with `...`):

```text
[fingerprint=0xb3fd16ac743048d0]
# n_ctx=65536 n_seq_max=1 kv_unified=0 n_threads=8 fa=auto type_k=1 type_v=1 strategy=auto predictor=15 profile=a44fff1f655de0d5

[variant budget=8000 cache_ubatch=8192 switch_mb=529.2 attn_frac=0.0755 head_mb=417.2 pcie=44.3 mtp_head_cpu=0 mtp_head_extra_mb=0 union_mb=7944 kernel_cap_mb=16 machine=8dcdfcc59bb70c60]
[tier 0 bs=1]
strategy=STATIC_ATTNPRIO_ALLMODELS n_pinned=10 n_attn_pinned=40 overflow=NONE tps=48.12 vram=7908.6 output_on_gpu=1 pin_from_back=0 overlap=1 switch_ms=0.00 ids_cross=0 slice_tokens=-1
ot=^output=CUDA_Host:0,^token_embd=CUDA_Host:3,blk\.0\..*=CUDA_Host:0, ... ,blk\.9\..*=CUDA_Host:0,blk\.10\.ffn_((up|gate|down)\.|(up|down|gate|gate_up)_(ch|)exps).*=CUDA_Host:3,blk\.10\..*=CUDA_Host:0, ... ,blk\.39\..*=CUDA_Host:0
cand strategy=GPUONLY_LAYERPIN_LAYERSTREAM viable=1 tps=2.71 n_pinned=12 n_attn_pinned=0 slots=0 vram=7813.8
cand strategy=STATIC_ATTNPRIO_ALLMODELS viable=1 tps=48.12 n_pinned=10 n_attn_pinned=40 slots=0 vram=7908.6
cand strategy=HYBRID_ATTNPRIO_FFNBALANCE viable=1 tps=48.12 n_pinned=10 n_attn_pinned=40 slots=0 vram=7908.6
[tier 1 bs=16]
...
[tier 5 bs=8192]
strategy=HYBRID_ATTNPRIO_FFNBALANCE n_pinned=4 n_attn_pinned=40 overflow=NONE tps=3050.93 vram=7696.4 output_on_gpu=1 pin_from_back=0 overlap=2 switch_ms=0.00 ids_cross=0 ffn_gpu=36 slice_tokens=450 piece=1:425.627,2:426.808, ... ,4096:1334.157
ot=^output=CUDA_Host:0,^token_embd=CUDA_Host:3,blk\.0\..*=CUDA_Host:0, ... ,blk\.4\.ffn_((up|gate|down)\.|(up|down|gate|gate_up)_(ch|)exps).*=CUDA_Host:1,blk\.4\..*=CUDA_Host:0,blk\.5\.ffn_((up|gate|down)\.|(up|down|gate|gate_up)_(ch|)exps).*=CUDA_Host:2,blk\.5\..*=CUDA_Host:0, ...
cand ...
```

An `EXPERT_POOL` tier adds the pool fields (this one from an older `PSHARD_STRATEGY=EXPERT_POOL` plan of the same model at `-c 512`, predictor 14):

```text
[tier 1 bs=16]
strategy=EXPERT_POOL n_pinned=0 n_attn_pinned=40 overflow=NONE tps=335.92 vram=8000.0 output_on_gpu=1 pin_from_back=0 overlap=1 switch_ms=32.03 ids_cross=1 K=0 s=80 miss_policy=hybrid prefill_mode=ab_stream hybrid_frac=0.485 cold_ms=32.03 slice_tokens=-1 piece=1:11.937,2:17.375,4:22.712,8:32.476
ot=^output=CUDA_Host:0,^token_embd=CUDA_Host:3,blk\.0\.ffn_(up|down|gate|gate_up)_exps\..*=CUDA_Host:1,blk\.0\..*=CUDA_Host:0,blk\.1\.ffn_(up|down|gate|gate_up)_exps\..*=CUDA_Host:2,blk\.1\..*=CUDA_Host:0, ...
```

## Planning for llama-bench

Use `--bench-plan` when the registry is intended for `llama-bench -pshard`. The planner accepts the bench shape flags `-p` / `--n-prompt`, `-n` / `--n-gen`, `-pg`, and `-d` / `--n-depth`, then plans each unique context that llama-bench can run:

```bash
./build/bin/llama-pshard-plan-params \
    --model /models/Qwen3.6-35B-A3B-Q8_0.gguf \
    --bench-plan \
    -pg 512,200 \
    -pg 2048,200 \
    -d 0,1024
```

For each generated context, `n_ctx` is the number of tokens that can be resident during the test:

- prompt-only: `n_ctx = p + d`
- generation-only: `n_ctx = n + d`
- prompt+generation: `n_ctx = p + n + d`

The largest planned tier for a bench context is capped by prompt batch demand, not by depth. For example, `-pg 2048,200 -d 1024` plans `n_ctx=3272` with `tier_cap=2048`, so the registry includes decode tiers plus prompt tiers up to `bs=2048`.

If `-fa` / `--flash-attn` is not provided and `LLAMA_ARG_FLASH_ATTN` is not set, `--bench-plan` uses Flash Attention off to match llama-bench defaults. Normal non-bench planning keeps the regular common parameter defaults.

`llama-bench -pshard` also plans every test configuration it has no plan for before the first test, so planning never lands between timed tests.

## Strategies

In the strategy names, `ATTN` refers to the attention/dense side of the layer, as opposed to FFN/MoE weights.

The planner prices every allowed strategy at each tier and keeps the viable plan with the lowest mean predicted time over the ubatch sizes the tier runs (its `piece=` curve), else the highest predicted tokens/s. `PSHARD_STRATEGY` sets which strategies are allowed:

- unset (the default ladder, pshard-1.0): `GPUONLY_LAYERPIN_LAYERSTREAM`, `STATIC_ATTNPRIO_ALLMODELS`, `HYBRID_ATTNPRIO_FFNBALANCE`.
- `ALL` (pshard-2.0): the default ladder plus `EXPERT_POOL`.
- a strategy name, or its id (`0` `GPUONLY_LAYERPIN_LAYERSTREAM`, `1` `STATIC_ATTNPRIO_ALLMODELS`, `2` `HYBRID_ATTNPRIO_FFNBALANCE`, `3` `EXPERT_POOL`): that strategy only (QA). A tier the forced strategy cannot fit keeps a `STATIC_ATTNPRIO_ALLMODELS` plan when one fits.

An invalid value is ignored with a warning. The registry header records the setting (`strategy=auto`, `ALL` or the name), and the fingerprint carries it: the runtime must see the same `PSHARD_STRATEGY` as the planner.

Static schedules run GPU-resident tensors on GPU and CPU-resident tensors on CPU, with no streamed GPU execution for host-resident weights.

- `STATIC_ATTNPRIO_ALLMODELS`: static attention-priority placement. It pins the attention/dense side across as many layers as fit, then uses the remaining budget to pin full layers. FFN/MoE that does not fit remains on CPU. Unlike `llama_params_fit`, this attention-priority placement applies to dense models too.

Dynamic schedules split the layer between CPU and GPU execution. Some host-resident tensors are streamed to GPU scratch for execution, while other parts of the layer remain on CPU.

- `HYBRID_ATTNPRIO_FFNBALANCE`: pin the attention/dense side across as many layers as fit, then full layers, and stream the remaining attention/dense side for GPU execution. The unpinned FFN/MoE is split between CPU execution and GPU streaming: `ffn_gpu` of them stream, spread evenly, the count with the lowest predicted CPU time plus GPU FFN time plus copy time not hidden behind other work (`ffn_gpu=0` keeps them all on CPU). The search runs three transports (two slots, one slot with copy-ahead, one slot without prefetch) and keeps the fastest. With the attention side pinned for every layer and every unpinned FFN/MoE streamed, the whole repeating-layer compute runs on GPU.

GPU-only schedules execute repeating-layer compute on GPU. Weights that do not fit in VRAM stay resident in host memory and are streamed to GPU scratch before use.

- `GPUONLY_LAYERPIN_LAYERSTREAM`: pin as many full layers as fit, then stream the remaining layers for GPU execution.

The pooled schedule (MoE models only) turns the routed experts into a VRAM cache.

- `EXPERT_POOL`: pin every layer except its routed experts, and the output head. The routed experts stay in host memory; the rest of the budget becomes a pool of `s` expert slots in every MoE layer, so a pool tier reports the whole budget as `vram`. A tier with `bs x n_expert_used < n_expert` is a cache tier: the slots keep the most recently used experts, and a miss follows the tier's miss policy. A larger tier is a whole-stack tier: each layer's experts stream through two alternating halves of the pool, hidden under the matrix multiplications (`prefill_mode=ab_stream`); the pool must hold two layers' experts.

### EXPERT_POOL miss policies

For each cache tier the planner prices four miss policies and keeps the fastest:

- `fetch`: copy the missed expert into the least recently used slot and compute it on GPU. The pool must hold a slot for every expert the pass can route to (`bs x n_expert_used`, at most `n_expert`).
- `cpu_admit`: compute the miss on CPU in this pass while its rows upload into a slot for the next pass.
- `hybrid`: fetch a share of the misses (`hybrid_frac`) and compute the rest on CPU, both at once.
- `cpu_exec`: compute every miss on CPU from host memory. The pool is never filled.

The other three need only one slot per layer. `fetch_on_2nd_miss` (first miss on CPU, a repeated miss fetches) is not a candidate; it runs only when forced. `PSHARD_MISS_POLICY=<name or id>` (`second` = `fetch_on_2nd_miss`) forces one policy for every cache tier (QA); when set, it is part of the fingerprint.

The planner always writes `prefill_mode=ab_stream`. `cpu_tail` (stream the stack, compute the ubatch's coldest experts on CPU) is defined but not chosen. `K` (layers whose experts are all resident) is always `0`.

A cache tier's hit rate comes from the routing workload (see [Inputs](#inputs)). `cold_ms` is the estimated extra decode time of its misses after it lands with empty slots.

The planner refuses `EXPERT_POOL` tiers when the GGUF has no routed-expert tensors, when there is no routing workload (no file and a failed calibration), or when the CPU profile lacks a pool entry (`PCIe_Segs_Kernel`, `PCIe_Segs_Kernel_Idle`, `DRAM_BW`, the `MUL_MAT` batch entries, `Pool_Serve_us`, `Pool_Split_us`). `llama-profiler-cpu --splice cpu_profile.txt` refreshes the profile header and keeps its op tables.

## Inputs

Machine profiles: `cpu_profile.txt` and `gpu_profile.txt` from `llama-profiler-cpu` and `llama-profiler-gpu`, read from the working directory or from the paths in `PSHARD_CPU_PROFILE` / `PSHARD_GPU_PROFILE`. The planner refuses to plan when the CPU profile is missing, predates schema 2, or was measured on another GPU, CPU, OS or thread count. There is no override. On a refusal the tool writes nothing for that context and exits with code `2`. A run with `-pshard` that is refused continues on the stock path and logs `>>> pshard DISABLED: this run uses the STOCK path ...`.

Routing workload (MoE models): `<model>.gguf.pshard_workload` holds per-layer route counts of the router's expert choices and the Zipf exponent fit to them (`zipf_alpha`, the input of the `EXPERT_POOL` hit-rate model). The expert pool adds its cache-tier routes at exit, so runs accumulate. Without the file, the planner calibrates with a CPU-only sampled generation (256 tokens, seed 1234, temperature 1) and saves it; the first real run replaces that calibration. `<model>.gguf.pshard_workload_rows` holds the distinct experts of multi-row MoE passes whose expert copies were sliced by used ids, added by pshard runs at exit. The planner prices sliced expert copies, `MUL_MAT_ID` expert reads and the pool hits and misses of multi-row passes from it. The ledgers are not part of the fingerprint: re-run the planner to price new ledger data into an existing plan.

## Runtime tier choice

The runtime loads the variant planned for its budget. Without one it uses a smaller budget's variant (the largest `cache_ubatch` first, then the largest budget) and reports a miss, so the run plans its own budget unless `--pshard-no-plan-on-miss` is set. It never uses a variant planned for a larger budget. It runs with `n_batch = n_ubatch = cache_ubatch`.

For each decode call of more than one token, the runtime prices the ways to cut the batch: one ubatch on a tier that holds it, or whole ubatches of one tier and the remainder on that tier or a smaller one. A ubatch costs its tier's `piece=` curve value. A plan switch costs the weights the new plan adds to the device and the KV/recurrent state that changes residency, at the variant's `pcie` rate, plus `cold_ms` when it lands an `EXPERT_POOL` cache tier, unless it comes from a cache tier with the same slot count. Switches are priced from the active plan, between the two tiers, and back to the `bs=1` plan. The cheapest cut runs and is logged:

```text
decode: pshard cut: 3402 tokens = 1 x bs=2048 + 1354 on bs=2048, priced 1395.6 ms
```

A single-token decode runs on the smallest viable tier that holds it.

## Flags

- `-pshard`: turn pshard on in the tools that load models through common (`llama-completion`, `llama-server`, `llama-perplexity`, ...). `llama-bench` has its own `-pshard` and `-mva`. The plan tool does not need it.
- `-mva` / `--max-vram-alloc <MiB>`: the budget, device memory for weights, KV and compute. If it is `0` or omitted, the budget is `--fit-budget` / `-fitb` when set, else the device's free VRAM minus `--fit-target` / `-fitt` (default 1024 MiB). `-fitt` is ignored when `-mva` or `-fitb` is set.
- `--pshard-tier-max <N>`: caps the largest tier. The default is `min(max(n_batch, 16384), n_ctx)`.
- `--pshard-no-plan-on-miss`: a run with no plan for its configuration, or only a smaller budget's, does not plan in-process. It falls back to the stock path, or keeps the smaller budget's plan.
- `--bench-plan` (plan tool only): see [Planning for llama-bench](#planning-for-llama-bench).
- The plan tool parses the speculative flags (`--spec-type`, `-md`, `--spec-draft-n-max`, ...). Pass the run's own: the fingerprint carries the MTP head and the verify tier.

pshard turns itself off when `-ngl`, `-ot`, `-ts`, `-sm row` / `-sm tensor` or `--no-kv-offload` is set.

## Notes

- Tiers: decode tiers `bs=1` and `bs=16`, plus a speculative verify tier of `n_draft + 1`. With several sequences: `1`, `n_parallel`, `16` (when `n_parallel < 16`) and `n_parallel x (n_draft + 1)`. Prefill tiers: `512`, `2048`, `4096`, `8192` up to the tier cap, and the cap itself when it is below 8192.
- The `[fingerprint=...]` line invalidates the registry when plan-compatible inputs change (`n_ctx`, `n_seq_max`, `kv_unified`, threads, FA mode, KV cache types, GGUF file size, whether the MTP head is loaded, the output caps (total and per sequence), `PSHARD_STRATEGY`, and `PSHARD_MISS_POLICY` when set), when the predictor's pricing model version changes (`predictor=`), or when the profile files change (`profile=` is a hash of `cpu_profile.txt` and `gpu_profile.txt`). The runtime must see the same profile files as the planner to match a plan. The comment below the fingerprint lists most of these inputs.
- Without speculative flags, the plan tool plans with one output per sequence, the cap `llama-completion -pshard` and `llama-server` run with. A tool with another cap (`llama-perplexity` keeps every token as an output) does not match that plan and plans its own configuration on load.
- Variant header fields: `budget` (MiB) and `cache_ubatch` key the variant. `switch_mb`, `attn_frac`, `head_mb` and `pcie` are the switch price inputs (MB of one layer, the attention share of it, MB of the output head, upload GB/s). `mtp_head_cpu=1` means the MTP head was moved to CPU so the union of the tiers fits the budget; `mtp_head_extra_mb` is retired and kept for parsing. `union_mb` is the largest device footprint of any tier and sizes the arena (a variant with a pool tier takes the whole budget). `kernel_cap_mb` is the profile's kernel-copy crossover: while a plan is active, pinned uploads up to this size run as copy kernels. `machine` is a hash of the GPU, CPU and OS; the runtime ignores a variant planned on another machine.
- Tier line fields: `n_pinned` whole layers, `n_attn_pinned` attention-side layers, `overflow` how the next layer is partly pinned (`NONE`; `ATTN`, `UP`, `GATE` stream its FFN, its gate and down, or its down; `MOE` streams its routed experts), `tps` predicted tokens/s at the tier's batch, `vram` MiB, `overlap` the transport (`1` two slots, `2` one slot with copy-ahead, `0` one slot without prefetch), `switch_ms` the estimated cost of switching into the plan from the `bs=1` plan, `ids_cross=1` when expert ids cross a split boundary so expert copies are sliced to the used experts, `slice_tokens` (ubatches below it copy only the used experts; `-1` = the scheduler's rule), and `piece=n:ms,...` the predicted time of one ubatch of `n` tokens below the tier's batch. `[tier ...] not_viable` marks a tier no strategy fits.
- `cand ...` lines after a tier's `ot=` line record every strategy the sweep priced for that tier (the pick and its runner-ups: viability, predicted tokens/s, pin counts, pool slots, device bytes). They are an audit trail only; the runtime never executes them, and older parsers skip them.
- A fingerprint can contain multiple `[variant budget=... cache_ubatch=...]` blocks. Re-running the planner with a new budget or cache ubatch replaces only that variant and keeps the others. The file keeps at most 32 fingerprints, and 16 variants per fingerprint; the oldest go first.
- `budget` is the arena budget. DeepSeek-V4 keeps its compressor state outside the arena, so its variant budget is `-mva` minus that state.
- One budget: with a separate draft model or an MTP context, the planner takes their device memory out of the budget.
- `cache_ubatch` records the runtime ubatch used for context/KV/SWA cache sizing. Tier compute scratch is still measured with the tier batch size.
- `pshard_disabled=1 baseline_vram=<MiB>` is variant-scoped. The planner writes it when the whole model fits the budget at the caller's batch sizes (the check is skipped when `PSHARD_STRATEGY` names one strategy); the tool then exits with code `2`, as no pshard plan was produced. Runtime skips pshard only when the measured baseline VRAM fits the current budget.
- `backend_id` values:
  - `0` = GPU pinned compute
  - `1` / `2` = shard compute lanes used for pipeline overlap
  - `3` = CPU
