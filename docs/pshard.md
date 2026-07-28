# Pipeline Sharding (pshard)

pshard enables model inference at user-specified VRAM budget.

The overflow weights are spilled to the pinned host memory which are streamed into device memory (VRAM) for GPU execution during runtime as per different strategies.

An offline planner probes candidate placements (host/device) against measured hardware characteristics and writes plans for different tiers; the runtime loads the appropriate plan. This gives three stages:


| Stage   | Tool                                       | Artifact                                        | Scope                      |
| ------- | ------------------------------------------ | ----------------------------------------------- | -------------------------- |
| Profile | `llama-profiler-gpu`, `llama-profiler-cpu` | `gpu_profile.txt`, `cpu_profile.txt`            | per machine                |
| Plan    | `llama-fit-params --pshard`                | `<model>.gguf.tensor_overrides.pshard_registry` | per model + context config |
| Run     | `llama-cli`, `llama-server`, `llama-bench` | —                                               | per invocation             |


A plan holds one entry per batch-size *tier*; the runtime selects the smallest tier covering
the current batch. Each entry is a set of tensor→backend assignments: GPU-resident, one of two
shard lanes used for pipelining, or host-resident.

## Limitations

- Single CUDA device. Multi-GPU is not supported.
- Architectures using the DSA or DSV4 KV cache (DeepSeek family) are not supported; pshard
currently declines without a diagnostic.
- Host RAM must hold the full model.



## Build

Standard CUDA build — see [docs/build.md](build.md). The pshard tools are part of the default
target set.

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

---



## 1. Profile the machine

The planner's cost model needs measured bandwidth and throughput, not vendor specs. The
profilers write `gpu_profile.txt` and `cpu_profile.txt` to the working directory, and the
planner resolves them from *its* working directory — run both from the same place.

```bash
./build/bin/llama-profiler-gpu
./build/bin/llama-profiler-cpu
```

`--fast` (default) sweeps a reduced configuration set; `--full` is exhaustive and considerably
slower. `--output` redirects, in which case point the planner at the results via
`PSHARD_CPU_PROFILE` / `PSHARD_GPU_PROFILE`.

The CPU profiler measures concurrent DRAM and PCIe bandwidth under load, so it takes a few
minutes. Results characterise the machine and are model-independent — generate once.

## 2. Plan a placement

```bash
./build/bin/llama-fit-params --pshard \
    --model models/your-model.gguf \
    --max-vram-alloc 8000 \
    -c 8192 -t 8 -fa on
```

The planner probes each supported strategy across the tier range, estimates throughput per
candidate from the profile data, and keeps the best viable plan per tier. Output is a plain-text
registry written next to the model.

`--max-vram-alloc` sets the planning budget in MiB. Omitted, the planner uses VRAM free at
planning time minus `--fit-target`, which makes the resulting plan dependent on whatever else
was resident on the device. Set it explicitly for reproducible plans.

Without profile data the planner degrades to VRAM-fit selection only, ignoring throughput. It
still produces a usable plan, but not necessarily the fastest one.

Strategy descriptions, the registry format, and planning for `llama-bench` shapes are covered
in [tools/fit-params/README-pshard.md](../tools/fit-params/README-pshard.md).

## 3. Run

```bash
./build/bin/llama-cli --pshard \
    --model models/your-model.gguf \
    --max-vram-alloc 8000 \
    -c 8192 -t 8 -fa on
```

`llama-server` and `llama-bench` accept the same flags.

### Plan/runtime fingerprint

The registry is keyed by a hash of the parameters that affect plan validity:

```
n_ctx  n_seq_max  n_threads  flash_attn_type  type_k  type_v  model_file_size  $PSHARD_STRATEGY
```

If the runtime's values hash differently there is no matching entry, and **pshard disables
itself and proceeds with a conventional load** — which then requires full VRAM. The fallback is
logged at warning level only, so it is easy to miss.

Consequently `-c`, `-np`, `-t`, `-fa`, `-ctk`, `-ctv` and the model must be identical between
planning and running. `-t` is the most common oversight, since its default is derived and
varies by tool; pass it explicitly to both.

`--max-vram-alloc` is deliberately excluded from the fingerprint. A single fingerprint can hold
several budget variants, and re-planning at a new budget replaces only that variant.

### Confirming pshard is active

```
common_init_result: pshard enabled, probing and loading plan cache
llama_params_fit_pshard_inference: loaded 7 tier plans from cache (variant budget=8000 MiB ...)
common_init_result: pshard runtime batch/ubatch set to selected cache_ubatch=8192
```

`no matching plan cache ... disabling pshard` or `pshard not active for this configuration`
indicates the fallback described above. `llama-bench` reports state directly in its `psh`
column.

---



## See also

- [docs/build.md](build.md)
- [tools/fit-params/README-pshard.md](../tools/fit-params/README-pshard.md) — planner
reference: strategies, registry format, `--bench-plan`

