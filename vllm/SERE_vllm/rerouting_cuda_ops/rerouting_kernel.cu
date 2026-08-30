#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>


// ---------------------------------------------------------------------------
// Fused kernel: clear high_mask + set bits + reroute in ONE launch.
// Must run as a single block so __syncthreads() provides the needed barriers.
// Handles any batch via a grid-stride-style loop within the single block.
// Suitable for decode phase (num_tokens <= ~256); falls back to original for
// larger batches (see fused_reroute_cuda).
// ---------------------------------------------------------------------------
template<typename scalar_t>
__global__ void fused_reroute_kernel(
    const int64_t* __restrict__ topk_ids,
    const scalar_t* __restrict__ similarity_matrix,
    int64_t* __restrict__ output_ids,       // write-back to topk_ids in-place
    const int num_tokens,
    const int top_k,
    const int select_top_k,
    const int num_experts,
    const scalar_t threshold) {

    extern __shared__ bool smem_high_mask[];   // num_experts bytes

    const int tid = threadIdx.x;
    const int bsz = blockDim.x;

    // Phase 1: clear shared high_mask
    for (int i = tid; i < num_experts; i += bsz)
        smem_high_mask[i] = false;
    __syncthreads();

    // Phase 2: mark high experts (first select_top_k slots of each token)
    const int set_elements = num_tokens * select_top_k;
    for (int i = tid; i < set_elements; i += bsz) {
        const int tok  = i / select_top_k;
        const int slot = i % select_top_k;
        const int64_t eid = topk_ids[tok * top_k + slot];
        if (eid >= 0 && eid < num_experts)
            smem_high_mask[eid] = true;
    }
    __syncthreads();

    // Phase 3: reroute the remaining slots
    const int reroute_slots    = top_k - select_top_k;
    const int reroute_elements = num_tokens * reroute_slots;

    for (int i = tid; i < reroute_elements; i += bsz) {
        const int tok          = i / reroute_slots;
        const int slot_in_rer  = i % reroute_slots;
        const int expert_slot  = select_top_k + slot_in_rer;

        const int64_t orig = topk_ids[tok * top_k + expert_slot];

        if (orig < 0 || orig >= num_experts) {
            output_ids[tok * top_k + expert_slot] = 0;
            continue;
        }

        // If this expert is already a high expert, keep it unchanged
        if (smem_high_mask[orig]) {
            output_ids[tok * top_k + expert_slot] = orig;
            continue;
        }

        scalar_t best_sim = static_cast<scalar_t>(-1e9);
        int      best_exp = 0;

        const scalar_t* row = similarity_matrix + orig * num_experts;
        for (int e = 0; e < num_experts; ++e) {
            if (smem_high_mask[e]) {
                const scalar_t s = row[e];
                if (s > best_sim) { best_sim = s; best_exp = e; }
            }
        }

        output_ids[tok * top_k + expert_slot] =
            (threshold > static_cast<scalar_t>(0.0) && best_sim < threshold)
            ? orig : static_cast<int64_t>(best_exp);
    }
}


// ---------------------------------------------------------------------------
// Original multi-block kernel (kept as fallback for large prefill batches).
// ---------------------------------------------------------------------------
template<typename scalar_t>
__global__ void reroute_kernel(
    const int64_t* __restrict__ topk_ids,
    const scalar_t* __restrict__ similarity_matrix,
    const bool* __restrict__ high_mask,
    int64_t* __restrict__ output_ids,
    const int num_tokens,
    const int top_k,
    const int select_top_k,
    const int num_experts,
    const scalar_t threshold) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int reroute_slots = top_k - select_top_k;
    if (reroute_slots <= 0) return;

    int token_idx    = idx / reroute_slots;
    int slot_in_rer  = idx % reroute_slots;
    int expert_slot  = select_top_k + slot_in_rer;

    if (token_idx >= num_tokens || expert_slot >= top_k) return;

    int input_idx           = token_idx * top_k + expert_slot;
    int64_t original_expert = topk_ids[input_idx];

    if (original_expert < 0 || original_expert >= num_experts) {
        output_ids[input_idx] = 0;
        return;
    }

    if (high_mask[original_expert]) {
        output_ids[input_idx] = original_expert;
        return;
    }

    scalar_t best_similarity = static_cast<scalar_t>(-1e9);
    int best_expert = 0;

    for (int high_expert = 0; high_expert < num_experts; high_expert++) {
        if (high_mask[high_expert]) {
            scalar_t similarity = similarity_matrix[original_expert * num_experts + high_expert];
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_expert     = high_expert;
            }
        }
    }

    if (threshold > 0.0 && best_similarity < threshold)
        output_ids[input_idx] = original_expert;
    else
        output_ids[input_idx] = best_expert;
}


// ---------------------------------------------------------------------------
// Fused entry point (1 kernel launch for decode, 3-op fallback for prefill).
// Threshold: if num_tokens * top_k > FUSED_MAX_WORK → use original path.
// ---------------------------------------------------------------------------
static constexpr int FUSED_MAX_TOKENS = 256;   // covers all practical decode batches

torch::Tensor fused_reroute_cuda(
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    torch::Tensor similarity_matrix,
    int64_t select_top_k,
    torch::Tensor high_mask_cache,
    torch::Tensor expert_mapping_cache,
    double threshold) {

    const int num_tokens = topk_weights.size(0);
    const int top_k      = topk_weights.size(1);
    const int num_experts = similarity_matrix.size(0);

    if (select_top_k <= 0 || select_top_k >= top_k)
        return topk_ids;

    const at::cuda::OptionalCUDAGuard device_guard(topk_ids.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (num_tokens <= FUSED_MAX_TOKENS) {
        // Single-block fused kernel: zero kernel-launch overhead
        const int set_elements     = num_tokens * (int)select_top_k;
        const int reroute_elements = num_tokens * (top_k - (int)select_top_k);
        const int threads = std::min(256,
                            std::max({num_experts, set_elements, reroute_elements}));
        const int smem_bytes = num_experts * (int)sizeof(bool);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, similarity_matrix.scalar_type(),
            "fused_reroute_kernel", [&] {
                fused_reroute_kernel<scalar_t><<<1, threads, smem_bytes, stream>>>(
                    topk_ids.data_ptr<int64_t>(),
                    similarity_matrix.data_ptr<scalar_t>(),
                    topk_ids.data_ptr<int64_t>(),
                    num_tokens, top_k, (int)select_top_k, num_experts,
                    static_cast<scalar_t>(threshold));
            });
    } else {
        // Fallback: original 3-op path for large prefill batches
        torch::Tensor high_mask;
        if (high_mask_cache.defined() &&
            high_mask_cache.numel() == num_experts &&
            high_mask_cache.device() == topk_weights.device()) {
            high_mask = high_mask_cache;
            high_mask.zero_();
        } else {
            high_mask = torch::zeros({num_experts},
                            torch::dtype(torch::kBool).device(topk_weights.device()));
        }

        auto high_experts = topk_ids.slice(1, 0, select_top_k);
        high_mask.scatter_(0, high_experts.flatten(), true);

        const int reroute_elements = num_tokens * (top_k - (int)select_top_k);
        const int threads_per_block = 256;
        const int blocks = (reroute_elements + threads_per_block - 1) / threads_per_block;

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kHalf, at::kBFloat16, similarity_matrix.scalar_type(),
            "reroute_kernel", [&] {
                reroute_kernel<scalar_t><<<blocks, threads_per_block, 0, stream>>>(
                    topk_ids.data_ptr<int64_t>(),
                    similarity_matrix.data_ptr<scalar_t>(),
                    high_mask.data_ptr<bool>(),
                    topk_ids.data_ptr<int64_t>(),
                    num_tokens, top_k, (int)select_top_k, num_experts,
                    static_cast<scalar_t>(threshold));
            });
    }

    return topk_ids;
}


// ---------------------------------------------------------------------------
// Legacy entry point (preserved for backward compatibility).
// ---------------------------------------------------------------------------
torch::Tensor reroute_cuda(
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    torch::Tensor similarity_matrix,
    int64_t select_top_k,
    torch::Tensor high_mask_cache,
    torch::Tensor expert_mapping_cache,
    double threshold) {

    // Delegate to fused path
    return fused_reroute_cuda(topk_weights, topk_ids, similarity_matrix,
                               select_top_k, high_mask_cache,
                               expert_mapping_cache, threshold);
}
