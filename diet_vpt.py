import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from torchvision.transforms.v2 import (
    RandomResizedCrop,
    ToDtype,
    Normalize,
    Compose,
    ColorJitter,
    PILToTensor,
)
from torch.utils.tensorboard import SummaryWriter
from vpt import VPTModel


class Food101Index(Food101):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, index


def main(
    batch_size=48,
    num_workers=48,
    epochs=10,
    learning_rate_backbone=0.001,
    learning_rate_W=0.0001,
    weight_decay_backbone=0.05,
    weight_decay_W=0.05,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
):
    torch.manual_seed(seed)

    model_name = "vit_base_patch14_reg4_dinov2.lvd142m"
    model = VPTModel(model_name=model_name, num_prompt_tokens=8)
    model.to(device)
    # Get model-specific transform (hard code it)
    # data_config = timm.data.resolve_data_config({}, model=model.base_model)
    # transform = timm.data.create_transform(**data_config, is_training=False)
    train_transforms = Compose(
        [
            RandomResizedCrop(size=518),
            PILToTensor(),
            ColorJitter(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )  # missing rotations and flips, will do them in gpu

    # Load the data
    train_dataset = Food101Index(
        root="data/food101", split="train", download=True, transform=train_transforms
    )

    # Create DataLoader for batching
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # Initialize the model
    input_size = model.base_model.embed_dim
    number_of_indices = len(train_dataset)
    # self supervised projector
    W = torch.nn.Linear(input_size, number_of_indices, bias=False).to(device)
    # superivsed head (for eval while training)
    sup_W = torch.nn.Linear(input_size, len(train_dataset.classes)).to(device)

    # Define loss function and optimizer
    criterion_ssl = nn.CrossEntropyLoss(label_smoothing=0.8)
    criterion_sup = nn.CrossEntropyLoss()
    backbone_optimizer = optim.AdamW(
        model.get_trainable_params(),
        lr=learning_rate_backbone,
        weight_decay=weight_decay_backbone,
    )
    len_train_loader = len(train_loader)
    backbone_scheduler = optim.lr_scheduler.OneCycleLR(
        backbone_optimizer,
        max_lr=learning_rate_backbone,
        total_steps=len_train_loader * epochs,
        pct_start=0.05,
    )
    sup_head_optimizer = optim.Adam(
        sup_W.parameters(), lr=learning_rate_W, weight_decay=weight_decay_W
    )
    W_optimizer = optim.Adam(
        W.parameters(), lr=learning_rate_W, weight_decay=weight_decay_W
    )

    # Training loop
    writer = SummaryWriter()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_sup_loss = 0.0
        for step, (images, labels, target_indices) in enumerate(train_loader):
            # zero gradients
            backbone_optimizer.zero_grad()
            W_optimizer.zero_grad()
            # prepare images
            images = images.to(device, non_blocking=True)
            B = images.shape[0]
            images[: B // 2] = images[: B // 2].flip(3)  # horizontal flip
            for quarter in range(4):  # 4 rotations
                images[(B // 4) * quarter : (B // 4) * (quarter + 1)] = torch.rot90(
                    images[(B // 4) * quarter : (B // 4) * (quarter + 1)],
                    quarter,
                    (2, 3),
                )
            # prepare targets
            target_indices = target_indices.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # --- Self-supervised step ---
            features = model(images)
            pred_indices = W(features)
            loss = criterion_ssl(pred_indices, target_indices.view(-1))
            loss.backward()
            backbone_optimizer.step()
            W_optimizer.step()
            backbone_scheduler.step()

            # --- Supervised step (monitor only, don't update backbone) ---
            # Detach features so gradients don't flow into backbone
            sup_head_optimizer.zero_grad()
            pred_labels = sup_W(features.detach())
            sup_loss = criterion_sup(pred_labels, labels)
            sup_loss.backward()
            sup_head_optimizer.step()

            # Logging
            loss_item = loss.item()
            sup_loss_item = sup_loss.item()
            running_loss += loss_item
            running_sup_loss += sup_loss_item
            print(
                f"Epoch {epoch+1}/{epochs}, Step {step}/{len_train_loader}, "
                f"SelfSupLoss={loss.item():.6f}, SupLoss={sup_loss.item():.6f}",
                end="\r",
            )
            writer.add_scalar(
                "loss/ssl", loss_item, global_step=step + epoch * len_train_loader
            )
            writer.add_scalar(
                "loss/sup", sup_loss_item, global_step=step + epoch * len_train_loader
            )
            current_lr = backbone_optimizer.param_groups[0]["lr"]
            writer.add_scalar('lr/backbone', current_lr, global_step=step + epoch * len_train_loader)
        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"SelfSupLoss: {running_loss/len_train_loader:.4f}, "
            f"SupLoss: {running_sup_loss/len_train_loader:.4f}"
        )

    # save the projector
    torch.save(model.state_dict(), "final_backbone.pth")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
