# Building a SERE venv + rerouting kernel

Everything needed to get from a bare GPU box to a Python environment that can
serve a SERE-rerouted model. Calibration (producing `similarity_matrices.pt`)
and the eval protocol are a separate document — see `calibration/README.md` and
`SERE_SETUP_RUNBOOK.md` in `hqta1110/moe-eval-unified`
(branch `feat/offline-batch-pinned-eval`).

## The one structural fact

**SERE is not a single build.** The serve side is a monkeypatch against a
specific vLLM MoE layer, so each model family needs its own plugin tree, its own
venv, and — for gemma-4 — a different vLLM major version.

| family | models | branch | vLLM | transformers | torch |
|---|---|---|---|---|---|
| `qwen` | Qwen3-30B-A3B, Qwen3.6-35B-A3B | `accuracy-bench-repro` | 0.18.1 | 4.57.6 | 2.10.0 |
| `glm` | GLM-4.7-Flash | `glm-4.7-flash` | 0.18.1 | 4.57.6 | 2.10.0 |
| `gemma4` | gemma-4-26B-A4B-it | `gemma4-support` | 0.29.0 | 5.17.0 | 2.13.0 |

The three branches carry **different `vllm/SERE_vllm/vllm_v1_patch.py` files**.
They are not interchangeable and must not be merged into one venv.

Check which tree a venv is actually bound to — an editable install points at a
directory, and it is easy to end up serving a different tree than you edited:

```bash
<venv>/bin/python -c "import SERE_vllm, os; print(os.path.dirname(SERE_vllm.__file__))"
```

## Prerequisites

- Linux + NVIDIA GPU. The kernel is compiled for the local arch at install time.
- CUDA toolkit with **`nvcc` on `PATH`**, matching the torch build in the table.
- [`uv`](https://docs.astral.sh/uv/).
- ~25 GB free for one venv (torch + vLLM wheels dominate).

**Never copy a built `.so` between machines.** Rebuild. A `.so` from another
box either fails to load or, worse, loads and misbehaves.

## Build

```bash
git clone https://github.com/hqta1110/SERE.git && cd SERE
git checkout accuracy-bench-repro     # or glm-4.7-flash, or gemma4-support
scripts/build_venv.sh --family qwen   # or --family glm / --family gemma4
```

Roughly 10–20 min, most of it wheel downloads; the kernel itself is 3–6 min.
It ends with `=== BUILD_OK ... ===` and prints the vLLM version and plugin path.

Flags: `--venv DIR` (default `./.venv`), and for GLM `--glm-shim-dir DIR`
(default `./.glm_shim`).

### What the script does

1. **Python stack.** `qwen`/`glm` sync from `repro/envs/v1/uv.lock` — a real lock,
   so the resolution is exact. `gemma4` installs from
   `repro/envs/gemma4/requirements.txt` — pinned versions frozen off a working
   box, reproducible but not hash-pinned (no lock exists for this stack).
2. **Kernel.** `uv pip install --no-build-isolation -e ./vllm`.
   `--no-build-isolation` is required: `setup.py` imports torch at build time,
   and an isolated build environment has no torch in it.
3. **GLM only:** installs a `glm4_moe_lite` config shim via a `.pth` file.
   transformers 4.57.6 has no such `model_type`, so `AutoConfig` cannot load the
   checkpoint without it.
4. **Verify** (see below).

## Verifying the build

The script asserts both of these, but they are worth knowing by hand.

**Plugin entry point** — vLLM discovers SERE through it:

```bash
<venv>/bin/python -c "
import importlib.metadata as m
print([e.name for e in m.entry_points(group='vllm.general_plugins')])"
# must contain: register_SERE_vllm
```

**The kernel — note the import path:**

```bash
<venv>/bin/python -c "from SERE_vllm.rerouting_cuda_ops import rerouting_ops_cuda; print('OK')"
```

> It is a **submodule**. `import rerouting_ops_cuda` at top level fails on a
> perfectly good install. That false negative has cost real hours here — if you
> are about to conclude "the kernel is missing", check the path first.

A missing kernel **hard-fails at serve time** rather than silently falling back.
`SERE_ALLOW_TORCH_REROUTE=1` forces a pure-torch path instead; it is for
debugging only and is much slower — never benchmark speed with it set, and
beware of it lingering in an exported shell environment.

## Confirming SERE actually rerouted anything

A clean build that reroutes nothing looks exactly like a successful run. Set
`SERE_COUNT_REROUTE=1` and read the line the plugin prints at process exit:

```
SERE_REROUTE_TOTALS calls=827820 slots=390387360 changed=85893790 changed_frac=0.2200
```

`changed_frac=0.0000` means SERE loaded but did nothing — treat it as a failure.

> **Trap:** the line is emitted once per process, and processes that never ran a
> MoE forward (the parent, a scorer) print `0.0000` — often *last*. Take the
> **max** across the log, never `tail -1`.

## Layout

```
BUILD.md                      this file
scripts/build_venv.sh         the builder
repro/envs/v1/                 uv.lock + pyproject for the vLLM 0.18.1 stack (qwen, glm)
repro/envs/gemma4/             frozen requirements for the vLLM 0.29.0 stack (gemma4)
vllm/                         the plugin + CUDA kernel; editable-installed
  SERE_vllm/vllm_v1_patch.py    <- the per-family file; differs across branches
  SERE_vllm/rerouting_cuda_ops/ rerouting_kernel.cu, rerouting_ops.cpp
calibration/                  HF-side adapters used to produce similarity_matrices.pt
  cal_expert_similarity.py      dispatcher: qwen2_moe, qwen3_moe, qwen3_5_moe,
                                deepseek_v2, gemma4, glm4_moe_lite
```

`calibration/` is family-independent and is identical on every branch — it runs
against plain transformers, not vLLM, so one calibration venv serves all models.
