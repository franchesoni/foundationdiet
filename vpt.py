import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import timm
import math


# Helper function for initialization (similar to the provided code)
def _xavier_uniform_init(tensor):
    """Applies Xavier uniform initialization to the tensor."""
    bound = math.sqrt(6.0 / tensor.shape[-1])
    nn.init.uniform_(tensor.data, -bound, bound)


class VPTModel(nn.Module):
    """
    Implements Deep Visual Prompt Tuning (VPT) for a generic timm Vision Transformer.
    """

    def __init__(self, model_name: str, num_prompt_tokens: int = 10):
        """
        Args:
            model_name (str): Name of the timm model to load (e.g., 'vit_base_patch16_224', 'dinov2_base').
            num_prompt_tokens (int): Number of prompt tokens to insert at each layer.
        """
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens
        # Load the pre-trained base model
        assert "dinov2" in model_name, "Only supports dinov2"
        self.base_model = timm.create_model(model_name, pretrained=True)
        # --- Get model specifics ---
        self.embed_dim = self.base_model.embed_dim
        # Determine the depth (number of transformer blocks)
        self.depth = len(self.base_model.blocks)
        # --- Initialize Prompt Tokens ---
        self.prompt_tokens = nn.Parameter(
            torch.zeros(self.depth, self.num_prompt_tokens, self.embed_dim)
        )
        _xavier_uniform_init(self.prompt_tokens)
        self.freeze()

    def freeze(self):
        """Freezes the parameters of the base model."""
        for param_name, param in self.base_model.named_parameters():
            param.requires_grad = False

        # Ensure prompts and new head are trainable
        self.prompt_tokens.requires_grad = True

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through patch embedding, position embedding, and transformer blocks with deep prompts."""
        B = x.shape[0]
        x = self.base_model.patch_embed(x)
        # Add position embedding
        x = self.base_model._pos_embed(x)  # Use timm's internal method for robustness
        x = self.base_model.patch_drop(x)
        x = self.base_model.norm_pre(x)

        # --- Deep Prompt Insertion through Blocks ---
        for i in range(len(self.base_model.blocks)):
            prompt_tokens_layer = self.prompt_tokens[i].unsqueeze(0).expand(B, -1, -1)
            x = torch.cat((x, prompt_tokens_layer), dim=1)
            num_tokens = x.shape[1]
            x = self.base_model.blocks[i](x)
            x = x[:, : num_tokens - self.num_prompt_tokens]  # Remove prompts

        x = self.base_model.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)[:, 0]

    def get_trainable_params(self):
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"- {name} (shape={param.shape}, numel={param.numel()})")
                trainable_params.append(param)
        return trainable_params

# --- Example Usage ---
if __name__ == "__main__":
    # Configuration
    model_name = (
        "vit_base_patch14_reg4_dinov2.lvd142m"  # Or 'vit_base_patch16_224', etc.
    )
    num_classes = 0  # Example: 100 classes for the downstream task
    num_prompt_tokens = 10  # Number of deep prompt tokens per layer
    img_size = 518  # Input image size

    # Create the VPT model
    vpt_model = VPTModel(
        model_name=model_name,
        num_classes=num_classes,
        num_prompt_tokens=num_prompt_tokens,
    )

    # Freeze the backbone
    print(f"Model: {model_name} with Deep VPT ({num_prompt_tokens} tokens/layer)")

    # --- Verify Trainable Parameters ---
    print("\nTrainable parameters:")
    trainable_params = vpt_model.get_trainable_params()
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params)}")

    # --- Example Forward Pass ---
    print("\nTesting forward pass...")
    dummy_input = torch.randn(4, 3, img_size, img_size)  # Batch size 4
    output = vpt_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Forward pass successful!")

    # --- Optimizer Setup (Example) ---
    # When training, only pass the trainable parameters to the optimizer
    # optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    # print("\nOptimizer would target parameters listed above.")
