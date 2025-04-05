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
    part,
    device="cuda",
):
    part_number, total_parts = map(int, part.split("."))  # Convert to integers
    part_number = part_number - 1
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

        # Split the dataset into parts
        total_samples = len(dataset)
        samples_per_part = total_samples // total_parts
        start_idx = part_number * samples_per_part
        end_idx = (
            start_idx + samples_per_part
            if part_number < total_parts - 1
            else total_samples
        )

        # Subset the dataset for this part
        subset_indices = list(range(start_idx, end_idx))
        subset = torch.utils.data.Subset(dataset, subset_indices)

        dataloader = DataLoader(
            subset,
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
        feature_file = (
            output_dir
            / f"food101_{split}_part{part_number+1}_of_{total_parts}_features.npy"
        )
        labels_file = (
            output_dir
            / f"food101_{split}_part{part_number+1}_of_{total_parts}_labels.npy"
        )
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

        print(
            f"Saving {len(all_features)} features for {split} set, part {part_number+1}..."
        )
        np.save(feature_file, all_features)
        np.save(labels_file, all_labels)


def unify_features(part, output):
    _, total_parts = map(int, part.split("."))  # Convert to integers
    for array_name in ["features", "labels"]:
        for split in ["train", "test"]:
            data_list = []
            for part in range(1, total_parts + 1):
                filepath = (
                    Path(output)
                    / f"food101_{split}_part{part}_of_{total_parts}_{array_name}.npy"
                )
                data = np.load(filepath)
                data_list.append(data)
            new_data = np.concatenate(data_list, axis=0)
            outpath = Path(output) / f"food101_{split}_{array_name}.npy"
            np.save(outpath, new_data)


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
    parser.add_argument(
        "--part", type=str, default="1.1", help="Process part n of m with notation n.m"
    )
    parser.add_argument(
        "--unify", action="store_true", help="Put parts in single files"
    )

    args = parser.parse_args()

    if args.unify:
        unify_features(part=args.part, output=args.output)
    else:
        extract_features(
            output_dir=args.output,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            part=args.part,
            device=args.device,
        )
