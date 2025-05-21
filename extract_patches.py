from pathlib import Path
import h5py
import openslide
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
from typing import Callable
from functools import partial
import argparse
import PIL.Image

NUM_BINS = 10


def random_sample(h5_path: Path, num_samples: int) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        coords = torch.tensor(f["coords"][:], dtype=torch.long)
        bytecounts = torch.tensor(f["bytecounts"][:], dtype=torch.float)
    if bytecounts.ndim > 1:
        bytecounts = torch.mean(bytecounts, dim=1)

    indices = torch.randint(0, len(coords), (num_samples,))
    sampled_coords = coords[indices]
    sampled_bytecounts = bytecounts[indices]

    # Sort the sampled coords by descending bytecount
    sorted_order = torch.argsort(sampled_bytecounts, descending=False)
    sorted_coords = sampled_coords[sorted_order]

    return sorted_coords


def entropy_bin_sampling(
    num_bins: int, h5_path: Path, num_samples: int, ignore_k_bins: int = 0
) -> torch.Tensor:
    if num_bins <= ignore_k_bins:
        raise ValueError(
            f"num_bins ({num_bins}) must be greater than ignore_k_bins ({ignore_k_bins})"
        )
    keep_k_bins = num_bins - ignore_k_bins

    with h5py.File(h5_path, "r") as f:
        coords = torch.tensor(f["coords"][:], dtype=torch.long)
        bytecounts = torch.tensor(f["bytecounts"][:], dtype=torch.float)
    if bytecounts.shape[0] != coords.shape[0]:
        raise ValueError("coords and bytecounts must have the same length")
    if bytecounts.ndim > 1:
        bytecounts = torch.mean(bytecounts, dim=1)

    bin_edges = torch.linspace(bytecounts.min(), bytecounts.max(), num_bins + 1)
    bin_indices = torch.bucketize(bytecounts, bin_edges) - 1
    max_bin = num_bins - 1
    min_bin = max(0, max_bin - keep_k_bins + 1)
    active_bins = list(range(min_bin, max_bin + 1))

    # Collect candidate indices
    all_candidates = []
    per_bin_target = num_samples // keep_k_bins
    for i in active_bins:
        bin_indices_i = torch.where(bin_indices == i)[0]
        if len(bin_indices_i) == 0:
            continue
        if len(bin_indices_i) < per_bin_target:
            sampled = bin_indices_i
        else:
            rand_idx = torch.randint(0, len(bin_indices_i), (per_bin_target,))
            sampled = bin_indices_i[rand_idx]
        all_candidates.append(sampled)

    if len(all_candidates) == 0:
        raise ValueError("No samples available in the top-k bins.")

    gathered = torch.cat(all_candidates)

    # Top up if we’re still short
    if len(gathered) < num_samples:
        remaining = num_samples - len(gathered)
        # gather all coords from active bins
        fallback_pool = torch.where(
            (bin_indices >= min_bin) & (bin_indices <= max_bin)
        )[0]
        fallback_pool = fallback_pool[~torch.isin(fallback_pool, gathered)]
        if len(fallback_pool) < remaining:
            raise ValueError(
                "Not enough fallback samples available to fulfill the request."
            )
        extra = fallback_pool[torch.randperm(len(fallback_pool))[:remaining]]
        gathered = torch.cat([gathered, extra])
    elif len(gathered) > num_samples:
        # trim to exact count
        gathered = gathered[torch.randperm(len(gathered))[:num_samples]]

    return coords[gathered]


def entropy_top_sampling(
    num_bins: int, h5_path: Path, num_samples: int, ignore_k_bins: int = 0
) -> torch.Tensor:
    if num_bins <= ignore_k_bins:
        raise ValueError(
            f"num_bins ({num_bins}) must be greater than ignore_k_bins ({ignore_k_bins})"
        )
    keep_k_bins = num_bins - ignore_k_bins

    with h5py.File(h5_path, "r") as f:
        coords = torch.tensor(f["coords"][:], dtype=torch.long)
        bytecounts = torch.tensor(f["bytecounts"][:], dtype=torch.float)

    if bytecounts.shape[0] != coords.shape[0]:
        raise ValueError("coords and bytecounts must have the same length")
    if bytecounts.ndim > 1:
        bytecounts = torch.mean(bytecounts, dim=1)

    bin_edges = torch.linspace(bytecounts.min(), bytecounts.max(), num_bins + 1)
    bin_indices = torch.bucketize(bytecounts, bin_edges) - 1
    max_bin = num_bins - 1
    min_bin = max(0, max_bin - keep_k_bins + 1)
    active_bins = list(range(min_bin, max_bin + 1))

    samples_per_bin = num_samples // keep_k_bins
    all_samples = []

    for bin_id in active_bins:
        bin_mask = bin_indices == bin_id
        bin_bytecounts = bytecounts[bin_mask]
        bin_coords = coords[bin_mask]

        if bin_bytecounts.numel() == 0:
            continue

        topk = min(samples_per_bin, bin_bytecounts.numel())
        sorted_indices = torch.argsort(bin_bytecounts, descending=True)[:topk]
        all_samples.append(bin_coords[sorted_indices])

    if not all_samples:
        raise ValueError("No samples found in selected bins.")

    gathered = torch.cat(all_samples)

    # If we have more or fewer than num_samples, adjust
    if len(gathered) > num_samples:
        gathered = gathered[torch.randperm(len(gathered))[:num_samples]]
    elif len(gathered) < num_samples:
        # fallback: top-N overall
        remaining = num_samples - len(gathered)
        fallback_mask = (bin_indices >= min_bin) & (bin_indices <= max_bin)
        fallback_indices = torch.where(fallback_mask)[0]
        fallback_bytecounts = bytecounts[fallback_indices]
        fallback_coords = coords[fallback_indices]

        already_used = torch.cat(
            [torch.nonzero(bin_indices == b).flatten() for b in active_bins]
        )
        fallback_indices = fallback_indices[~torch.isin(fallback_indices, already_used)]

        if fallback_indices.numel() < remaining:
            raise ValueError("Not enough fallback samples to complete the request.")

        sorted_fallback = torch.argsort(fallback_bytecounts, descending=True)[
            :remaining
        ]
        fallback_top_coords = fallback_coords[sorted_fallback]
        gathered = torch.cat([gathered, fallback_top_coords])

    return gathered


class PatchDataset(Dataset):
    def __init__(
        self,
        slide_path: Path,
        h5_path: Path,
        patch_size=512,
        sampling_func: Callable[[Path], torch.Tensor] = partial(
            random_sample, num_samples=1000
        ),
    ):
        self.slide = openslide.OpenSlide(slide_path)
        self.patch_size = patch_size
        self.coords = sampling_func(h5_path)

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        coord = self.coords[idx]
        x, y = coord[0], coord[1]
        patch = self.slide.read_region((x, y), 0, (self.patch_size, self.patch_size))
        patch = patch.convert("RGB")
        img = transforms.functional.to_image(patch)
        return img, coord


def extract_slide_patches(
    slide_path: Path,
    h5_path: Path,
    output_dir: Path,
    patch_size: int,
    sampling_func: Callable[[Path], torch.Tensor],
) -> None:
    dataset = PatchDataset(slide_path, h5_path, patch_size, sampling_func)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    for batch in dataloader:
        for i in range(batch[0].shape[0]):
            sample = batch[0][i]
            coord = batch[1][i]
            pil_img: PIL.Image.Image = transforms.functional.to_pil_image(sample)
            pil_img.save(
                output_dir / f"{slide_path.stem}/{coord[0]}_{coord[1]}.jpeg",
            )


def main(args):
    slide_dir = Path(args.slide_dir)
    slide_paths = list(slide_dir.glob("*.svs"))
    h5_dir = Path(args.h5_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_size = 512
    num_samples = args.num_samples

    if args.method == "random":
        sampling_func = partial(
            random_sample,
            num_samples=num_samples,
        )
    elif args.method == "entropy":
        sampling_func = partial(
            entropy_bin_sampling,
            num_samples=num_samples,
            num_bins=args.num_bins,
            ignore_k_bins=args.ignore_k_bins,
        )
    else:
        raise ValueError(f"Unknown sampling method: {args.method}")

    for slide_path in slide_paths:
        h5_path = h5_dir / f"{slide_path.stem}_patches.h5"
        dataset = PatchDataset(slide_path, h5_path, patch_size, sampling_func)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        for batch in dataloader:
            for i in range(batch[0].shape[0]):
                sample = batch[0][i]
                coord = batch[1][i]
                pil_img: PIL.Image.Image = transforms.functional.to_pil_image(sample)
                pil_img.save(
                    output_dir / f"{slide_path.stem}/{coord[0]}_{coord[1]}.jpeg",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slide_dir",
        type=str,
        required=True,
        help="Directory containing the slide files.",
    )
    parser.add_argument(
        "--h5_dir",
        type=str,
        required=True,
        help="Directory containing the h5 files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output patches.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for DataLoader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["random", "entropy"],
        default="random",
        help="Sampling method to use.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to extract from each slide.",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=NUM_BINS,
        help="Number of bins for entropy sampling.",
    )
    parser.add_argument(
        "--ignore_k_bins",
        type=int,
        default=0,
        help="Number of bins to ignore for entropy sampling.",
    )
    args = parser.parse_args()
    main(args)
    # Example usage:
    # python extract_patches.py --slide_dir /mnt/nfs03-R6/staining/TCGA_256_40 --h5_dir /mnt/nfs03-R6/staining/trident/40x_512px_0px_overlap/patches --output_dir /mnt/nfs03-R6/staining/tiles/random --method random --num_samples 1000 --batch_size 100 --num_workers 0
