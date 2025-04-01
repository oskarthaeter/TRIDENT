"""
Example usage:

```
python run_single_slide.py --slide_path output/wsis/394140.svs --job_dir output/ --mag 20 --patch_size 256
```

"""

import argparse
import os

from trident import OpenSlideWSI
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory


def parse_arguments():
    """
    Parse command-line arguments for processing a single WSI.
    """
    parser = argparse.ArgumentParser(description="Process a WSI from A to Z.")
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
    return parser.parse_args()


def process_slide(args):
    """
    Process a single WSI by performing segmentation, patch extraction, and feature extraction sequentially.
    """

    # Initialize the WSI
    print(f"Processing slide: {args.slide_path}")
    slide = OpenSlideWSI(
        slide_path=args.slide_path,
        lazy_init=False,
        custom_mpp_keys=args.custom_mpp_keys,
    )

    # Step 1: Tissue Segmentation
    print("Running tissue segmentation...")
    segmentation_model = segmentation_model_factory(
        model_name=args.segmenter,
        confidence_thresh=args.seg_conf_thresh,
        device=f"cuda:{args.gpu}",
    )
    slide.segment_tissue(
        segmentation_model=segmentation_model,
        target_mag=segmentation_model.target_mag,
        job_dir=args.job_dir,
    )
    print(
        f"Tissue segmentation completed. Results saved to {args.job_dir}contours_geojson and {args.job_dir}contours"
    )

    # Step 2: Tissue Coordinate Extraction (Patching)
    print("Extracting tissue coordinates...")
    save_coords = os.path.join(
        args.job_dir, f"{args.mag}x_{args.patch_size}px_{args.overlap}px_overlap"
    )

    coords_path = slide.extract_tissue_coords(
        target_mag=args.mag, patch_size=args.patch_size, save_coords=save_coords
    )
    print(f"Tissue coordinates extracted and saved to {coords_path}.")

    # Step 3: Visualize patching
    viz_coords_path = slide.visualize_coords(
        coords_path=coords_path,
        save_patch_viz=os.path.join(save_coords, "visualization"),
    )
    print(f"Tissue coordinates extracted and saved to {viz_coords_path}.")

    # Step 4: Feature Extraction
    print("Extracting features from patches...")
    encoder = encoder_factory(args.patch_encoder)
    encoder.eval()
    encoder.to(f"cuda:{args.gpu}")
    features_path = features_dir = os.path.join(
        save_coords, "features_{}".format(args.patch_encoder)
    )
    slide.extract_patch_features(
        patch_encoder=encoder,
        coords_path=os.path.join(save_coords, "patches", f"{slide.name}_patches.h5"),
        save_features=features_dir,
        device=f"cuda:{args.gpu}",
    )
    print(f"Feature extraction completed. Results saved to {features_path}")


def main():
    args = parse_arguments()
    process_slide(args)


if __name__ == "__main__":
    main()
