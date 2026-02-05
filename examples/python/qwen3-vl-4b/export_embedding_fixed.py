"""
Export a simple multimodal embedding model for Qwen3-VL that accepts vision_hidden_states.
This bypasses the GQA issues by directly wrapping the embedding layer.
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
import os

def export_multimodal_embedding():
    """Export embedding model that accepts both text and vision inputs."""
    
    # Load model
    pytorch_dir = "pytorch"
    model = AutoModel.from_pretrained(
        pytorch_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager"  # Force eager attention to avoid GQA export issues
    )
    model.eval()
    
    # Get embedding layer
    embed_tokens = model.model.language_model.embed_tokens
    hidden_size = embed_tokens.embedding_dim  # 2560
    
    class MultimodalEmbedding(nn.Module):
        def __init__(self, embed_layer, hidden_size, merge_size=2):
            super().__init__()
            self.embed_tokens = embed_layer
            self.hidden_size = hidden_size
            self.merge_size = merge_size
            # Vision start token ID for Qwen3-VL
            self.vision_start_token_id = 151652
        
        def forward(self, input_ids, vision_hidden_states):
            """
            Merge text embeddings with vision features.
            
            Args:
                input_ids: (batch, seq_len) - token IDs including vision_start tokens
                vision_hidden_states: (num_vision_tokens, hidden_size) - vision features
            
            Returns:
                inputs_embeds: (batch, seq_len, hidden_size) - merged embeddings
            """
            # Get text embeddings
            inputs_embeds = self.embed_tokens(input_ids)  # (batch, seq_len, hidden_size)
            
            # Find vision_start token positions
            batch_size, seq_len, hidden_dim = inputs_embeds.shape
            
            # Flatten for easier indexing
            inputs_embeds_flat = inputs_embeds.reshape(-1, hidden_dim)
            input_ids_flat = input_ids.reshape(-1)
            
            # Find indices where vision_start tokens appear
            vision_mask = (input_ids_flat == self.vision_start_token_id)
            vision_indices = vision_mask.nonzero(as_tuple=True)[0]
            
            # Calculate number of vision tokens after merge
            num_vision_features = vision_hidden_states.shape[0]
            num_vision_tokens = (num_vision_features + self.merge_size - 1) // self.merge_size
            
            # Replace vision_start tokens with actual vision features
            if len(vision_indices) > 0 and num_vision_features > 0:
                # Take only as many vision features as there are vision_start tokens
                num_to_replace = min(len(vision_indices), num_vision_features)
                for idx in range(num_to_replace):
                    pos = vision_indices[idx]
                    inputs_embeds_flat[pos] = vision_hidden_states[idx]
            
            # Reshape back
            inputs_embeds = inputs_embeds_flat.reshape(batch_size, seq_len, hidden_dim)
            
            return inputs_embeds
    
    # Create wrapper
    wrapper = MultimodalEmbedding(embed_tokens, hidden_size)
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 20
    num_vision_tokens = 144  # 576 patches / 4 (merge_size^2)
    
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    # Add a vision_start token
    dummy_input_ids[0, 5] = 151652
    
    dummy_vision_hidden_states = torch.randn(num_vision_tokens, hidden_size)
    
    # Export
    output_path = "../cpu-fp32/qwen3vl-embedding.onnx"
    
    print("Exporting multimodal embedding model...")
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_vision_hidden_states),
        output_path,
        input_names=["input_ids", "vision_hidden_states"],
        output_names=["inputs_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "vision_hidden_states": {0: "num_vision_tokens"},
            "inputs_embeds": {0: "batch", 1: "sequence"}
        },
        opset_version=17,
        do_constant_folding=False  # Keep it simple to avoid optimization issues
    )
    
    print(f"✓ Exported: {output_path}")
    
    # Verify
    import onnx
    model = onnx.load(output_path)
    print("\nModel inputs:")
    for inp in model.graph.input:
        print(f"  - {inp.name}")
    print("\nModel outputs:")
    for out in model.graph.output:
        print(f"  - {out.name}")

if __name__ == "__main__":
    export_multimodal_embedding()
