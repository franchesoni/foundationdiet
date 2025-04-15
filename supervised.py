from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def main(
    batch_size=48,
    num_workers=96,
    epochs=10,
    learning_rate=0.001,
    device="cuda" if torch.cuda.is_available() else "cpu",
    path_to_diet_checkpoint=None,
):
    # Load the data
    train_features = np.load(
        "/home/fmarchesoni/repos/foundationdiet/data/features/food101_train_features.npy"
    )
    train_labels = np.load(
        "/home/fmarchesoni/repos/foundationdiet/data/features/food101_train_labels.npy"
    )
    test_features = np.load(
        "/home/fmarchesoni/repos/foundationdiet/data/features/food101_test_features.npy"
    )
    test_labels = np.load(
        "/home/fmarchesoni/repos/foundationdiet/data/features/food101_test_labels.npy"
    )

    # Convert data to PyTorch tensors
    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Create DataLoader for batching
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    if path_to_diet_checkpoint is not None and Path(path_to_diet_checkpoint).exists():
        from diet_shallow import ShallowMLP

        diet_model = ShallowMLP(train_features.shape[1])
        diet_model.load_state_dict(
            torch.load(path_to_diet_checkpoint, map_location="cpu")
        )
        diet_model = diet_model.to(device)
        do_diet = True
    else:
        do_diet = False

    # Define the linear classifier
    class LinearClassifier(nn.Module):
        def __init__(self, input_size, num_classes):
            super(LinearClassifier, self).__init__()
            self.fc = nn.Linear(input_size, num_classes)

        def forward(self, x):
            return self.fc(x)

    # Initialize the model
    input_size = train_features.shape[1]
    num_classes = len(torch.unique(train_labels))
    model = LinearClassifier(input_size, num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if do_diet:
                features = diet_model(features)
            outputs = model(features)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(
                f"Epoch {epoch+1}/{epochs}, Step {step}/{len(train_loader)}", end="\r"
            )
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for step, (features, labels) in enumerate(test_loader):
            features, labels = features.to(device, non_blocking=True), labels.view(
                -1
            ).to(device, non_blocking=True)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f"Evaluating step {step}/{len(test_loader)}", end="\r")

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
