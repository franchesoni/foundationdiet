import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def reconstruct_indices(total_samples, total_parts, augmentations_count):
    """
    Reconstruct the original dataset indices and augmentation mappings.

    Args:
        total_samples (int): Total number of samples in the dataset.
        total_parts (int): Total number of parts the dataset was split into.
        augmentations_count (int): Number of augmentations applied.

    Returns:
        indices (list): List of original dataset indices.
        augmentations (list): List of augmentation indices corresponding to each feature.
    """
    indices = []
    augmentations = []

    samples_per_part = total_samples // total_parts

    for part_number in range(total_parts):
        # Calculate the start and end indices for this part
        start_idx = part_number * samples_per_part
        end_idx = (
            start_idx + samples_per_part
            if part_number < total_parts - 1
            else total_samples
        )
        part_indices = list(range(start_idx, end_idx))

        # Repeat the indices for each augmentation
        for aug_idx in range(augmentations_count):
            indices.extend(part_indices)
            augmentations.extend([aug_idx] * len(part_indices))

    return indices, augmentations


# Define the linear classifier
class ShallowMLP(nn.Module):
    def __init__(self, input_size, expansion_factor=4):
        super(ShallowMLP, self).__init__()
        self.l1 = nn.Linear(input_size, input_size * expansion_factor)
        self.l2 = nn.Linear(input_size * expansion_factor, input_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.l2(out) + x  # residual
        return out


def main(
    batch_size=48,
    num_workers=96,
    epochs=10,
    learning_rate=0.001,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=42,
):
    torch.manual_seed(seed)
    # Load the data
    train_features = np.load(
        "/home/fmarchesoni/repos/foundationdiet/data/features/food101_train_features.npy"
    )

    # Convert data to PyTorch tensors
    train_features = torch.tensor(train_features, dtype=torch.float32)
    indices = torch.tensor(
        reconstruct_indices(
            train_features.shape[0] // 8, total_parts=2, augmentations_count=8
        )[0]
    )

    # Create DataLoader for batching
    train_dataset = TensorDataset(train_features, indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # Initialize the model
    input_size = train_features.shape[1]
    number_of_indices = len(torch.unique(indices))
    model = ShallowMLP(input_size).to(device)
    W = torch.nn.Linear(input_size, number_of_indices, bias=False).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.8)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(W.parameters()), lr=learning_rate, weight_decay=0.05
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=len(train_loader) * epochs,
        pct_start=0.05,
    )

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = W(model(features))
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_item = loss.item()
            running_loss += loss_item
            print(
                f"Epoch {epoch+1}/{epochs}, Step {step}/{len(train_loader)}, Loss={loss_item:.6f}",
                end="\r",
            )
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # save the projector
    torch.save(model.state_dict(), "shallow_mlp.pth")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
