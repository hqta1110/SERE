# Reproducing ARES & SERE on GLM-4.7-Flash (another machine)

End-to-end recipe for the GLM-4.7-Flash (`zai-org/GLM-4.7-Flash`) accuracy runs:
**baseline · ARES (routing_mass_step) · SERE (rerouting)**. GLM-4.7-Flash is a
DeepSeek-style MoE — `Glm4MoeLiteForCausalLM` / `model_type=glm4_moe_lite`, **64
routed experts, top-4**, 1 shared expert, 1 dense layer (`first_k_dense_replace`),
sigmoid router with `e_score_correction_bias`, grouped-topk with **n_group=1**
(so grouping is trivial → plain top-k). 47 layers, ~30B bf16.

Everything here is isolated from the working qwen SERE setup (`SERE_v1` +
its venv are untouched). Paths below are this box's (`/home/PC/...`); adjust for yours.

---

## 0. Prerequisites (all machines)

- **GPUs:** serving = TP=2 (~30 GB/GPU, fits a 40 GB card); SERE calibration = the
  full 30B sharded across GPUs (`device_map=auto`, needs ~60 GB aggregate).
- **Tools:** `uv`, `nvcc` on PATH (kernels compile locally), an HF token.
- **CUDA kernels are sm-specific — ALWAYS rebuild locally, never copy a `.so`**
  (this box = A100 sm_80).
- **NCCL note (GCP only):** GCP images set `NCCL_NET=gIB` for multi-node GPUDirect,
  which breaks single-node TP. The serve scripts apply a repair (`NCCL_NET=Socket`,
  `NCCL_IB_DISABLE=1`, drop `NCCL_TUNER_CONFIG_PATH`, strip `gib` from
  `LD_LIBRARY_PATH`). On a normal machine this is a harmless no-op — keep it.

### 0a. Download the model
```bash
hf download zai-org/GLM-4.7-Flash --exclude "*.pth" "original/*" "*.gguf"
# note the snapshot dir it lands in; call it $SNAP below (~59 GB, bf16, 48 shards)
```

### 0b. The `glm4_moe_lite` config shim (BOTH methods need it)
vLLM 0.18.1 ships the `glm4_moe_lite` *model* but the pinned serve transformers
(4.57.6) does **not** ship its *config* → vLLM's AutoConfig validation fails. The
one-file shim `glm4_moe_lite_register.py` (in this repo) registers a
`Glm4MoeLiteConfig` (a `Glm4MoeConfig` subclass with `model_type="glm4_moe_lite"`;
vLLM's `Glm4MoeLite` is a bare subclass of `Glm4MoE`, so the config matches). Install
it into **every serve venv** as a `.pth` so the main process AND vLLM workers pick it up:
```bash
SP=$(<venv>/bin/python -c "import site;print(site.getsitepackages()[0])")
cp glm4_moe_lite_register.py /some/stable/dir/
printf '/some/stable/dir\nimport glm4_moe_lite_register\n' > "$SP/zz_glm_lite.pth"
# verify: <venv>/bin/python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('$SNAP').model_type)"  -> glm4_moe_lite
```

---

## 1. ARES on GLM-4.7-Flash

**Repo/branch (the fix matters):** `giangntt/new-efficient-moe @ feat/rms-rescue-no-remap`
(commit `2d5e052`). This branch has the **row-argmax decisiveness rescue** — required
for grouped-topk models: vLLM's `grouped_topk` returns *unsorted* weights, and the
old slot-0/slot-1 rescue would keep the wrong expert. (The fused `routing_mass_step`
kernel itself engages fine on grouped-topk; only the rescue needed the fix.)

```bash
git clone -b feat/rms-rescue-no-remap https://github.com/giangntt/new-efficient-moe.git
cd new-efficient-moe
uv venv --python 3.11 && uv sync
uv pip install --no-build-isolation ./patches/vllm/fused_skip_ops   # build fused kernel (~2 min)
# verify: .venv/bin/python -c "import sys;sys.path.insert(0,'.'); from patches.vllm import fused_skip_ops as f; print(f.routing_mass_step_available(), f.fused_confidence_available())"  -> True True
```
Then install the config shim (§0b) into this venv.

**Serve** (routing_mass_step; set `PYTHONPATH=<repo>` so `sitecustomize` patches the
workers). Config knobs are the operating point:
```bash
# NCCL repair (see §0), then:
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=<repo> \
  EXPERT_SKIP_MODE=dynamic \
  EXPERT_SKIP_ONLINE_THRESHOLD_METHOD=routing_mass_step \
  EXPERT_SKIP_ONLINE_CONTRIB_MASS_FRACTION=0.15 \   # c1: 0.15 ; c2/default: 0.2
  EXPERT_SKIP_ONLINE_MIN_ACTIVE_FRAC=0.0 \          # c1/c2: 0.0 ; default: 0.25
  EXPERT_SKIP_ONLINE_DECISIVENESS_MARGIN=0.1 \      # c1: 0.1 ; c2: 0.05
  EXPERT_SKIP_DISABLED_LAYERS=0,46 \                # GLM has 47 layers (layer 0 dense)
  EXPERT_SKIP_ONLINE_LOG=1 \
  <repo>/.venv/bin/vllm serve $SNAP --served-model-name glm-4.7-flash \
     --trust-remote-code --tensor-parallel-size 2 --max-model-len 16384 \
     --gpu-memory-utilization 0.90 --no-enable-prefix-caching --port 8001
```
Confirm the fused path engaged: the risk monitor emits `risk=...` lines (accumulated
inside the fused apply kernel). MoE backend must be **TRITON** (`Using TRITON backend
for Unquantized MoE` in the log) — GLM is bf16, so it is.

On this box the ready-made wrapper is `/home/PC/glm_serve_ares_c1.sh` (c1) — same env,
plus the NCCL repair.

---

## 2. SERE on GLM-4.7-Flash

**Isolated tree (this repo):** `SERE-glm`, branch `glm-support` — a copy of
`SERE_v1 @ e380fbc` (`hqta1110/SERE @ accuracy-bench-repro`) with the GLM additions.
Never touches the working `SERE_v1` / its venv. To recreate on another machine:
```bash
git clone -b accuracy-bench-repro https://github.com/hqta1110/SERE.git SERE-glm   # base
cd SERE-glm && git checkout -b glm-support
# apply the GLM diff (this repo's glm-support branch) — 2 files + 1 new adapter:
#   vllm/SERE_vllm/vllm_v1_patch.py           (+GLM block patch, sigmoid/scale-1.0 pins)
#   calibration/cal_expert_similarity.py      (+glm4_moe_lite dispatch/device_map/save)
#   calibration/adapted_modeling_glm4_moe_lite.py   (NEW: fused-expert calibration adapter)
```

### 2a. Serve venv (isolated) — `build_venv.sh`
Builds a dedicated venv from the vendored **v1 serve lock**
(`moe-eval-unified/repro/env/v1/{pyproject.toml,uv.lock}` — vLLM 0.18.1, torch 2.10,
tf 4.57.6), then the SERE rerouting kernel, then the glm shim:
```bash
bash build_venv.sh    # -> SERE-glm/.venv ; verifies register_SERE_vllm + kernel + glm config
```
(On another machine, point `SPEC` in the script at your copy of the v1 lock, and the
`.pth` dir at wherever you placed `glm4_moe_lite_register.py`.)

### 2b. Calibrate (produce `similarity_matrices.pt`) — `calibrate_glm.sh`
`glm4_moe_lite` isn't in tf 4.57.6, so calibration runs in a **tf ≥5.16** env
(here `reap/.venv_q36`). The GLM-Lite experts are **fused** (stacked `gate_up_proj`
/`down_proj`), so the adapter uses the qwen3_5 fused-einsum math. FineWeb-Edu, frobenius:
```bash
bash calibrate_glm.sh   # device_map=auto, batch_size 32 x max_len 512 -> ~4 GB transient/GPU
# -> artifacts/glm-4.7-flash-sere/similarity_matrices.pt  (46 MoE layers, [64,64] each)
```
Data: `SERE/calibration/data/fineweb_edu_calibration.parquet` (~3.6 MB; ship it).
**Memory gotcha:** the adapter materializes the unweighted `(E=64, T, H=2048)` expert
tensor per layer — keep `batch_size*max_len` modest (128×512 OOM'd a 40 GB shard; 32×512
is safe). More tokens → gentler batch on bigger cards.

### 2c. Serve SERE (injection path) — `serve_sere_glm.sh <gpus> <port> <select_top_k> <threshold>`
No baked checkpoint — the plugin injects the `.pt` per layer. Grouped-topk needs **no**
special code: n_group=1 ⇒ `fused_topk_bias` (sigmoid + bias + renormalize) reproduces
GLM routing; the GLM patch only pins `scoring_func="sigmoid"` and routing
`scale=1.0` (the block applies its `routed_scaling_factor`=1.8 to the MoE *output*).
```bash
bash serve_sere_glm.sh 0,1 8003 2 0.5    # SERE-tuned: select_top_k=2, threshold=0.5
bash serve_sere_glm.sh 2,3 8004 1 0.0    # SERE-aggr:  select_top_k=1, threshold=0.0
# confirm the GLM patch fired: log has 46x "Enabled SERE routing (injected) for Glm4MoeLite layer N"
```

---

## 3. Run the accuracy eval (both methods)

Uses the `moe-eval-unified` harness against a running server. Pass `--tokenizer $SNAP`
(the served name `glm-4.7-flash` is not an HF id):
```bash
cd moe-eval-unified
.venv/bin/python tools/run_eval_client.py --task gsm8k \
   --method <name> --model glm-4.7-flash --base-url http://localhost:<port>/v1 --concurrency 64
# tasks: gsm8k, math_hard, math500, aime, humaneval_plus, mbppplus, lcb
# scores -> results/glm-4.7-flash/<method>/<ts>/<task>/scores.json
```
For a `vllm bench serve` speed A/B, also pass `--tokenizer $SNAP`.

---

## 4. Results measured on this box (seed-42, GLM-4.7-Flash)

| method | gsm8k | math_hard |
|---|---|---|
| vanilla | 85.5 | 38.9 |
| ARES c1 (mass .15/floor .0) | 81.7 (−3.8) | 32.2 (−6.6) |
| SERE-tuned (k2, thr .5) | 84.8 (−0.8) | 39.1 (+0.2) |
| SERE-aggr (k1, thr 0) | 85.3 (−0.2) | 34.0 (−4.9) |

Reading: ARES (deletion) pays a real hard-task cost on this **top-4** model; SERE
(substitution) is near-lossless at the tuned setting — but partly because, with mean
expert similarity ~0.17, few reroutes clear threshold 0.5, so SERE-tuned barely acts
(little speedup). See the analysis notes in the session log.

## 5. Machine-specific bits to re-point on another host
- All `/home/PC/...` paths (snapshot dir, repos, venvs, the shim dir, FineWeb parquet).
- The GLM snapshot **hash** in the `*.sh` scripts (yours will differ) — set `$SNAP`.
- `build_venv.sh` `SPEC` → your copy of `moe-eval-unified/repro/env/v1`.
- Rebuild BOTH kernels locally (ARES `fused_skip_ops`, SERE `rerouting_ops`).
- Drop the NCCL-repair lines if not on GCP-gIB (harmless if kept).
