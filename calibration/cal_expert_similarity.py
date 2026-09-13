#!/usr/bin/env python3
"""
Expert similarity calculation
"""

import argparse
import os
import gc
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoConfig
# Adapters are imported lazily per model_type: some (qwen3_moe, deepseek) use
# transformers APIs removed in newer versions, so importing them all at module load
# would break model_types (e.g. qwen3_5_moe) that need a transformers-main env.
from utils import print_similarity_statistics


def main():
    parser = argparse.ArgumentParser(description='Expert similarity calculation')
    parser.add_argument('--model_type', type=str, required=True, 
                       choices=['qwen2_moe', 'qwen3_moe', 'deepseek_v2', 'qwen3_5_moe', 'gemma4', 'glm4_moe_lite'],
                       help='Model type')
    parser.add_argument('--model_path', type=str, required=True, help='Original model path')
    parser.add_argument('--output_path', type=str, required=True, help='Output model path')
    parser.add_argument('--data_path', type=str, required=True, help='Calibration data path')
    parser.add_argument('--max_len', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--similarity_method', type=str, default='frobenius',
                       choices=['cka', 'cosine', 'frobenius'],
                       help='Similarity computation method')
    parser.add_argument('--kernel', type=str, default='linear',
                       choices=['linear', 'rbf', 'polynomial'],
                       help='CKA kernel type')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Expert Similarity Calculation System")
    print("=" * 60)
    print(f"Similarity method: {args.similarity_method}")
    print(f"Kernel: {args.kernel}")
    print(f"Batch size: {args.batch_size}")

    # Load data
    print("\nLoading calibration data...")
    df = pd.read_parquet(args.data_path)
    all_texts = df['text'].tolist()
    print(f"Total text samples: {len(all_texts)}")

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Filter texts with token length > max_len
    print(f"Filtering texts with token length > {args.max_len}...")
    texts = []
    for i, text in enumerate(all_texts):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) > args.max_len:
            texts.append(text)
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1}/{len(all_texts)}, filtered: {len(texts)}")

    print(f"Filtered text samples: {len(texts)} (rate: {len(texts)/len(all_texts)*100:.1f}%)")

    if len(texts) == 0:
        raise ValueError(f"No texts found with token length > {args.max_len}")

    print(f"Using {len(texts)} qualifying text samples")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    if args.model_type == "qwen2_moe":
        from adapted_modeling_qwen2_moe import Qwen2MoeForCausalLM
        model_class = Qwen2MoeForCausalLM
    elif args.model_type == "qwen3_moe":
        from adapted_modeling_qwen3_moe import Qwen3MoeForCausalLM
        model_class = Qwen3MoeForCausalLM
    elif args.model_type == "deepseek_v2":
        from adapted_modeling_deepseek import DeepseekV2ForCausalLM
        model_class = DeepseekV2ForCausalLM
    elif args.model_type == "qwen3_5_moe":
        from adapted_modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
        model_class = Qwen3_5MoeForCausalLM
    elif args.model_type == "gemma4":
        from adapted_modeling_gemma4 import Gemma4ForCausalLM
        model_class = Gemma4ForCausalLM
    elif args.model_type == "glm4_moe_lite":
        from adapted_modeling_glm4_moe_lite import Glm4MoeLiteForCausalLM
        model_class = Glm4MoeLiteForCausalLM
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}!")

    # qwen3.5-moe is a ~64GB multimodal model -> shard across GPUs.
    device_map = 'auto' if args.model_type in ("qwen3_5_moe", "gemma4", "glm4_moe_lite") else 'cuda'
    model = model_class.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True
    )
    model.eval()

    # The text decoder layers: qwen3.5-moe nests them under language_model.
    if args.model_type in ("qwen3_5_moe", "gemma4"):
        decoder_layers = model.model.language_model.layers
    else:
        decoder_layers = model.model.layers

    # Where the MoE lives on a decoder layer. Gemma4 has NO MoE block: the layer
    # owns `router` and `experts` directly, and `layer.mlp` is the parallel DENSE
    # MLP that runs alongside the MoE on every layer -- calibrating that would
    # silently calibrate the wrong module.
    moe_attr = 'experts' if args.model_type == "gemma4" else 'mlp'

    # Enable similarity computation for all MoE layers
    print("Configuring similarity computation...")
    moe_layers = []
    layer_indices = []
    for layer_idx, layer in enumerate(decoder_layers):
        moe = getattr(layer, moe_attr, None)
        if moe is not None and hasattr(moe, 'enable_similarity_computation'):
            moe.enable_similarity_computation(
                method=args.similarity_method,
                kernel=args.kernel
            )
            moe_layers.append(moe)
            layer_indices.append(layer_idx)

    print(f"Found {len(moe_layers)} MoE layers: {layer_indices}")

    print("Starting similarity calculation...")

    # Prepare input data
    batch_texts = texts[:args.batch_size]
    print(f"Using {len(batch_texts)} texts for similarity calculation")

    # Tokenize
    inputs = tokenizer(
        batch_texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=args.max_len
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print(f"Input shape: {inputs['input_ids'].shape}")

    # Reset similarity matrices
    for moe_layer in moe_layers:
        if hasattr(moe_layer, 'reset_similarity_matrix'):
            moe_layer.reset_similarity_matrix()

    # Run forward pass
    print("Running inference and computing similarity matrices...")
    with torch.no_grad():
        outputs = model(**inputs)

    print("Inference completed!")

    # Collect similarity matrices
    similarity_matrices = {}
    for i, (layer_idx, moe_layer) in enumerate(zip(layer_indices, moe_layers)):
        similarity_matrix = moe_layer.get_similarity_matrix()
        if similarity_matrix is not None:
            # Move to CPU to save GPU memory
            similarity_matrices[layer_idx] = similarity_matrix.clone().cpu()
            print(f"Layer {layer_idx}: similarity matrix shape {similarity_matrices[layer_idx].shape}")

    print(f"\nSimilarity calculation completed!")

    # Display statistics using utils function
    print_similarity_statistics(similarity_matrices, layer_indices)
    
    
    # Save model and similarity matrices
    print("\nSaving model...")
    os.makedirs(args.output_path, exist_ok=True)

    # Save model with similarity matrices. For qwen3_5_moe the SERE serve-side is a
    # plugin monkeypatch that loads similarity_matrices.pt into the BASE checkpoint,
    # so we skip the ~64GB full-model copy (disk) and only emit the matrices.
    if args.model_type not in ("qwen3_5_moe", "gemma4", "glm4_moe_lite"):
        model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    # Save similarity matrices
    similarity_save_path = os.path.join(args.output_path, 'similarity_matrices.pt')
    torch.save(similarity_matrices, similarity_save_path)

    print(f"Model saved to: {args.output_path}")
    print(f"Similarity matrices saved to: {similarity_save_path}")
    print(f"Saved similarity matrices for {len(similarity_matrices)} layers")

    # Display summary
    print("\n" + "=" * 60)
    print("Calculation completed!")
    print("=" * 60)
    

if __name__ == '__main__':
    main() 