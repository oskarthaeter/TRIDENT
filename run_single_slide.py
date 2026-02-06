"""
Example usage:

```
python run_single_slide.py --slide_path output/wsis/394140.svs --job_dir output/ --mag 20 --patch_size 256
```

"""

import argparse
import os
import torch
import time

from trident import OpenSlideWSI
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory


def parse_arguments():
    """
    Parse command-line arguments for processing a single WSI.
    """
    parser = argparse.ArgumentParser(description="Process a WSI from A to Z.")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Whether to run in benchmark mode",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU index to use for processing tasks"
    )
    parser.add_argument(
        "--slide_path", type=str, required=True, help="Path to the WSI file to process"
    )
    parser.add_argument(
        "--job_dir", type=str, required=True, help="Directory to store outputs"
    )
    parser.add_argument(
        "--patch_encoder",
        type=str,
        default="conch_v15",
        choices=[
            "conch_v1",
            "uni_v1",
            "uni_v2",
            "ctranspath",
            "phikon",
            "resnet50",
            "gigapath",
            "virchow",
            "virchow2",
            "hoptimus0",
            "hoptimus1",
            "h0mini",
            "phikon_v2",
            "conch_v15",
            "musk",
            "hibou_l",
            "kaiko-vits8",
            "kaiko-vits16",
            "kaiko-vitb8",
            "kaiko-vitb16",
            "kaiko-vitl14",
            "lunit-vits8",
        ],
        help="Patch encoder to use",
    )
    parser.add_argument(
        "--mag",
        type=int,
        choices=[5, 10, 20, 40],
        default=20,
        help="Magnification at which patches/features are extracted",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Patch size at which coords/features are extracted",
    )
    parser.add_argument(
        "--segmenter",
        type=str,
        default="hest",
        choices=[
            "hest",
            "grandqc",
            "entropy",
        ],
        help="Type of tissue vs background segmenter. Options are HEST or GrandQC.",
    )
    parser.add_argument(
        "--seg_conf_thresh",
        type=float,
        default=0.5,
        help="Confidence threshold to apply to binarize segmentation predictions. Lower this threhsold to retain more tissue. Defaults to 0.5. Try 0.4 as 2nd option.",
    )
    parser.add_argument(
        "--custom_mpp_keys",
        type=str,
        nargs="+",
        default=None,
        help="Custom keys used to store the resolution as MPP (micron per pixel) in your list of whole-slide image.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Absolute overlap for patching in pixels. Defaults to 0. ",
    )
    parser.add_argument(
        "--export_entropy_format",
        action="store_true",
        default=False,
        help="Whether to adjust coords so that tile-aligned dataloader can be used for patch extraction.",
    )
    parser.add_argument(
        "--k_random_patches",
        type=int,
        default=None,
        help="If set, randomly sample k patches from the extracted tissue coordinates for feature extraction.",
    )
    return parser.parse_args()


def process_slide(args):
    """
    Process a single WSI by performing segmentation, patch extraction, and feature extraction sequentially.
    """
    start_time = time.perf_counter()
    # Initialize the WSI
    # print(f"Processing slide: {args.slide_path}")
    torch.cuda.nvtx.range_push("Initialize WSI")
    slide = OpenSlideWSI(
        slide_path=args.slide_path,
        lazy_init=False,
        custom_mpp_keys=args.custom_mpp_keys,
    )
    torch.cuda.nvtx.range_pop()

    # Step 1: Tissue Segmentation
    # print("Running tissue segmentation...")
    exclude_segment_loading_duration = 0.0
    torch.cuda.nvtx.range_push("Tissue Segmentation")
    if args.segmenter == "entropy":
        slide.segment_tissue_alternative(
            holes_are_tissue=False,
            job_dir=args.job_dir,
        )
    else:
        exclude_segment_loading_start = time.perf_counter()
        segmentation_model = segmentation_model_factory(
            model_name=args.segmenter,
            confidence_thresh=args.seg_conf_thresh,
            device=f"cuda:{args.gpu}",
        )
        exclude_segment_loading_end = time.perf_counter()
        exclude_segment_loading_duration = (
            exclude_segment_loading_end - exclude_segment_loading_start
        )
        slide.segment_tissue(
            segmentation_model=segmentation_model,
            target_mag=segmentation_model.target_mag,
            job_dir=args.job_dir,
        )
    torch.cuda.nvtx.range_pop()
    # print(
    #     f"Tissue segmentation completed. Results saved to {args.job_dir}/contours_geojson and {args.job_dir}/contours"
    # )

    # Step 2: Tissue Coordinate Extraction (Patching)
    # print("Extracting tissue coordinates...")
    torch.cuda.nvtx.range_push("Tissue Coordinate Extraction")
    save_coords = os.path.join(
        args.job_dir, f"{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap"
    )

    coords_path = slide.extract_tissue_coords(
        target_mag=args.mag,
        patch_size=args.patch_size,
        save_coords=save_coords,
        export_entropy_format=args.export_entropy_format,
        k_random_patches=args.k_random_patches,
    )
    torch.cuda.nvtx.range_pop()
    # print(f"Tissue coordinates extracted and saved to {coords_path}.")

    # Step 3: Visualize patching
    if not args.benchmark:
        torch.cuda.nvtx.range_push("Visualize Patching")
        viz_coords_path = slide.visualize_coords(
            coords_path=coords_path,
            save_patch_viz=os.path.join(save_coords, "visualization"),
        )
        torch.cuda.nvtx.range_pop()
    # print(f"Tissue coordinates extracted and saved to {viz_coords_path}.")

    # Step 4: Feature Extraction
    # print("Extracting features from patches...")
    torch.cuda.nvtx.range_push("Feature Extraction")
    exclude_encoder_loading_start = time.perf_counter()
    encoder = encoder_factory(args.patch_encoder)
    encoder.eval()
    encoder.to(f"cuda:{args.gpu}")
    exclude_encoder_loading_end = time.perf_counter()
    exclude_encoder_loading_duration = (
        exclude_encoder_loading_end - exclude_encoder_loading_start
    )

    features_path = features_dir = os.path.join(
        save_coords, "features_{}".format(args.patch_encoder)
    )
    slide.extract_patch_features(
        patch_encoder=encoder,
        coords_path=os.path.join(save_coords, "patches", f"{slide.name}_patches.h5"),
        save_features=features_dir,
        device=f"cuda:{args.gpu}",
    )
    torch.cuda.nvtx.range_pop()
    # print(f"Feature extraction completed. Results saved to {features_path}")
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    effective_duration = (
        total_duration
        - exclude_segment_loading_duration
        - exclude_encoder_loading_duration
    )
    return effective_duration


def clear_job_dir(job_dir):
    if os.path.exists(job_dir):
        for root, dirs, files in os.walk(job_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)


def main():
    args = parse_arguments()
    # clear job dir
    if args.benchmark:
        n = 25
        # warm-up run
        process_slide(args)

        # timed runs
        durations = []
        for _ in range(n):
            clear_job_dir(args.job_dir)
            durations.append(process_slide(args))

        # print all durations in csv format
        print("Duration (seconds)")
        for i, duration in enumerate(durations):
            print(duration)
    else:
        process_slide(args)


if __name__ == "__main__":
    torch.cuda.cudart().cudaProfilerStart()
    main()
    torch.cuda.cudart().cudaProfilerStop()

    # python run_single_slide.py --slide_path /mnt/research2/oskar/TCGA/TCGA-CE-A3ME-01Z-00-DX1.BCF7C127-B617-43C3-9D54-E10310EE9DA5.svs --job_dir output/ --segmenter hest --mag 40 --patch_size 512 --patch_encoder h0mini
    ## 62.49 seconds
    # python run_single_slide.py --slide_path /mnt/research2/oskar/TCGA/TCGA-CE-A3ME-01Z-00-DX1.BCF7C127-B617-43C3-9D54-E10310EE9DA5.svs --job_dir output/ --segmenter hest --mag 40 --patch_size 512 --patch_encoder h0mini --k_random_patches 20
    ## 5.57 seconds
    # python run_single_slide.py --slide_path /mnt/research2/oskar/TCGA/TCGA-CE-A3ME-01Z-00-DX1.BCF7C127-B617-43C3-9D54-E10310EE9DA5.svs --job_dir output/ --segmenter entropy --export_entropy_format --mag 40 --patch_size 512 --patch_encoder h0mini
