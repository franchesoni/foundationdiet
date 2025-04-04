import torch
import timm
import tqdm
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from torchvision.transforms.v2 import functional as TF


def extract_features(
    output_dir,
    batch_size,
    num_workers,
    device="cuda",
):
    """Extract features from Food101 dataset with specified augmentations"""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use DINO model as in the original code
    model_name = "vit_base_patch14_reg4_dinov2.lvd142m"
    model = timm.create_model(
        model_name, num_classes=0, checkpoint_path="model.safetensors"
    )
    model.eval()
    device = device if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Get model-specific transform
    data_config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    # Process training set with augmentations and test set without
    for split in ["train", "test"]:
        print(f"Processing {split} set...")
        dataset = Food101(
            root="data/food101", split=split, download=True, transform=transform
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
        if split == "train":
            augmentations = [
                lambda x: x,  # original
                lambda x: TF.hflip(x),  # horizontal flip
                lambda x: TF.vflip(x),  # vertical flip
                lambda x: TF.vflip(TF.hflip(x)),  # both flips
                lambda x: TF.rotate(x, 90),  # 90° rotation
                lambda x: TF.rotate(x, 270),  # 270° rotation
                lambda x: TF.hflip(TF.rotate(x, 90)),  # 90° + horizontal flip
                lambda x: TF.hflip(TF.rotate(x, 270)),  # 270° + horizontal flip
            ]
        elif split == "test":
            augmentations = [lambda x: x]

        # Begin feature extraction
        feature_file = output_dir / f"food101_{split}_features.npy"
        labels_file = output_dir / f"food101_{split}_labels.npy"
        all_features = []
        all_labels = []

        for augmentation in augmentations:
            for batch in tqdm.tqdm(dataloader):
                imgs, labels = batch
                with torch.no_grad():
                    aimgs = augmentation(imgs.to(device, non_blocking=True))
                    feats = model.forward_features(aimgs)
                    cls_feats = feats[:, 0, :].cpu().numpy()
                    all_features.append(cls_feats)
                    all_labels.extend(labels.numpy())

        # Concatenate and save features
        all_features = np.vstack(all_features)
        all_labels = np.vstack(all_labels)
        print(f"Features shape: {all_features.shape}")
        print(f"Labels shape: {all_labels.shape}")

        print(f"Saving {len(all_features)} features for {split} set...")
        np.save(feature_file, all_features)
        np.save(labels_file, all_labels)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract features from Food101 dataset"
    )
    parser.add_argument(
        "--output", default="./data/features", help="Output directory for features"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )

    args = parser.parse_args()

    extract_features(
        output_dir=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
