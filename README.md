# 🔱   Trident

 [arXiv](https://arxiv.org/abs/2501.16652) | [Cite](https://github.com/mahmoodlab/trident?tab=readme-ov-file#reference) | [License](https://github.com/mahmoodlab/trident?tab=readme-ov-file#license-and-terms-of-tuse)


Trident is a toolkit for large-scale whole-slide image processing.
This project was developed by the [Mahmood Lab](https://faisal.ai/) at Harvard Medical School and Brigham and Women's Hospital.

> [!NOTE]
> Contributions are welcome! Please report any issues. You may also contribute by opening a pull request.

## Key Features:

<img align="right" src="_readme/trident_crop.png" width="250px" />

- **Tissue Segmentation**: Extract tissue from background using a DeepLabv3 model (supports H&E, IHC, penmark and artifact removal, etc.).
- **Patch Extraction**: Extract tissue patches of any size and magnification.
- **Patch Feature Extraction**: Extract patch embeddings from tissue patches using 13 popular foundation models, including [UNI](https://www.nature.com/articles/s41591-024-02857-3), [CONCH](https://www.nature.com/articles/s41591-024-02856-4), [Virchow](https://www.nature.com/articles/s41591-024-03141-0), [H-Optimus-0](https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0) and many more...
- **Slide Feature Extraction**: Extract slide embeddings from pre-extracted patch embeddings using 5 whole-slide foundation models, including [Titan](https://arxiv.org/abs/2411.19666), 
[PRISM](https://arxiv.org/abs/2405.10254), [GigaPath](https://www.nature.com/articles/s41586-024-07441-w) and [CHIEF](https://www.nature.com/articles/s41586-024-07894-z). 

## Getting Started:

### 🔨 1. **Installation**:
- Create a conda environment: `conda create -n "trident" python=3.10`, and activate it `conda activate trident`.
- **With cloning**:
    - `git clone git@github.com:mahmoodlab/trident.git && cd trident`.
    - Local install with running `pip install -e .`.
- **With pip**:
    - `pip install git+ssh://git@github.com:mahmoodlab/trident.git`
- Additional packages may be required if you are loading specific pretrained models. Follow error messages for additional instructions.

### 🔨 2. **Running Trident**:

**Are you experienced with WSI processing?**

Process a batch of WSIs for patch feature extraction at once with:

```
python run_batch_of_slides.py --task all --wsi_dir wsis --job_dir ./trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256
```

**Or follow step-by-step instructions:**

**Step 1: Tissue Segmentation**
 - **Description**: Segments tissue vs. background regions from a list of WSIs in `wsi_dir`.
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task seg --wsi_dir ./wsis --job_dir ./trident_processed --gpu 0
   ```
   - `--task seg`: Specifies that you want to do tissue segmentation.
   - `--wsi_dir ./wsis`: Path to the directory containing WSIs.
   - `--job_dir ./trident_processed`: Output directory for processed results.
   - `--gpu 0`: Uses GPU with index 0 for computation.
 - **Outputs**:
   - WSI thumbnails are saved in `./trident_processed/thumbnails`.
   - WSI thumbnails annotated with tissue contours are saved in `./trident_processed/contours`.
   - GeoJSON files containing tissue contours are saved in `./trident_processed/contours_geojson`. These can be opened in [QuPath](https://qupath.github.io/) for editing/quality control, if necessary.

 **Step 2: Tissue Patching**
 - **Description**: Extracts patches from segmented tissue regions at a specific magnification.
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task coords --wsi_dir wsis --job_dir ./trident_processed --mag 20 --patch_size 256 --overlap 0
   ```
   - `--task coords`: Specifies that you want to do patching.
   - `--wsi_dir wsis`: Path to the directory containing WSIs.
   - `--job_dir ./trident_processed`: Output directory for processed results.
   - `--mag 20`: Extracts patches at 20x magnification.
   - `--patch_size 256`: Each patch is 256x256 pixels.
   - `--overlap 0`: Patches overlap by 0 pixels. Note that this is the absolute overlap in pixels, i.e. use `--overlap 128` for 50% overlap on patches of size 256.
 - **Outputs**:
   - Patch coordinates are saved as h5 files in `./trident_processed/20x_256px/patches`.
   - WSI thumbnails annotated with patch borders are saved in `./trident_processed/20x_256px/visualization`.

 **Step 3a: Patch Feature Extraction**
 - **Description**: Extracts features from tissue patches using a specified encoder.
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task feat --wsi_dir wsis --job_dir ./trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256 
   ```
   - `--task feat`: Specifies that you want to do feature extraction.
   - `--wsi_dir wsis`: Path to the directory containing WSIs.
   - `--job_dir ./trident_processed`: Output directory for processed results.
   - `--patch_encoder uni_v1`: Uses the `UNI` patch encoder. Could also be `conch_v1`, `ctranspath`, `gigapath`, `virchow`, `hoptimus0`, etc. See below for list of supported models. 
   - `--mag 20`: Features are extracted from patches at 20x magnification.
   - `--patch_size 256`: Patches are 256x256 pixels in size.
 - **Outputs**: 
   - Features are saved as h5 files in `./trident_processed/20x_256px/features_uni_v1`. (Shape: `(n_patches, feature_dim)`)

Trident supports 13 patch encoders, loaded via our [`encoder_factory`](https://github.com/mahmoodlab/trident/blob/main/trident/patch_encoder_models/load.py#L11). Models requiring specific installations will return error messages with additional instructions. Gated models on HuggingFace require access requests: 

- **UNI**: [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI)  (`--patch_encoder uni_v1`)
- **UNIv2**: [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)  (`--patch_encoder uni_v2`)
- **CONCH**: [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH)  (`--patch_encoder conch_v1`)
- **CONCHv1.5**: [MahmoodLab/conchv1_5](https://huggingface.co/MahmoodLab/conchv1_5)  (`--patch_encoder conch_v15`)
- **Virchow**: [paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow)  (`--patch_encoder virchow`)
- **Virchow2**: [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2)  (`--patch_encoder virchow2`)
- **Phikon**: [owkin/phikon](https://huggingface.co/owkin/phikon)  (`--patch_encoder phikon`)
- **Phikon-v2**: [owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2/)  (`--patch_encoder phikon_v2`)
- **Prov-Gigapath**: [prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath)  (`--patch_encoder gigapath`)
- **H-Optimus-0**: [bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0)  (`--patch_encoder hoptimus0`)
- **MUSK**: [xiangjx/musk](https://huggingface.co/xiangjx/musk)  (`--patch_encoder musk`)
- **CTransPath**: Automatic download  (`--patch_encoder ctranspath`)
- **ResNet50**: Pretrained on ImageNet via torchvision.  (`--patch_encoder resnet50`)

**Step 3b: Slide Feature Extraction**
 - **Description**: Extracts slide embeddings using a specified slide encoder. If not pre-extracted, this command will automatically extract patch embeddings too. 
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task feat --wsi_dir wsis --job_dir ./trident_processed --slide_encoder titan --mag 20 --patch_size 512 
   ```
   - `--task feat`: Specifies that you want to do feature extraction.
   - `--wsi_dir wsis`: Path to the directory containing WSIs.
   - `--job_dir ./trident_processed`: Output directory for processed results.
   - `--slide_encoder titan`: Uses the `Titan` slide encoder. Could also be `prism`, `gigapath`, `chief`, and `threads` (soon!). 
   - `--mag 20`: Features are extracted from patches at 20x magnification.
   - `--patch_size 512`: Patches are 512x512 pixels in size.
 - **Outputs**: 
   - Features are saved as h5 files in `./trident_processed/20x_256px/slide_features_titan`. (Shape: `(feature_dim)`)

Supported models include:
- **Threads**: Coming Soon! [MahmoodLab/threads](https://huggingface.co/MahmoodLab/threads) (`--slide_encoder threads`).
- **Titan**: [MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN) (`--slide_encoder titan`)
- **PRISM**: [paige-ai/Prism](https://huggingface.co/paige-ai/Prism) (`--slide_encoder prism`)
- **CHIEF**: [CHIEF](https://github.com/hms-dbmi/CHIEF) (`--slide_encoder chief`)
- **GigaPath**: [prov-gigapath]()  (`--slide_encoder gigapath`)

**Note**: `trident` will embed each slide independently rather than in a patient-specific manner. This is because defining patient-specific embeddings requires additional inputs (such as a mapping between patient and slide). However, the goal is to provide quick and off-the-shelf slide embeddings. [patho-bench](https://github.com/mahmoodlab/Patho-Bench) can assist in extracting patient embeddings by aggregating multiple slides per patient.

Additional information is provided in `tutorials`.

**Feeling overwhelmed?**

Run the single slide processing script:

```
python run_single_slide.py --slide_path wsis/xxxx.svs --job_dir ./trident_processed --mag 20 --patch_size 256
```

## Quality Control

trident outputs a variety of files for quality control. It is recommended that you review these files after each step to ensure that the results are as expected.

1. Segmentation contours are saved in the `./<job_dir>/contours` directory. These are thumbnails of the WSI with the tissue contours drawn in green.

<img src="_readme/contours.png" alt="WSI thumbnail with the tissue contours drawn in green." height="150px">

2. Patch annotations are saved in the `./<job_dir>/<patch_dir>/visualization` directory. These are thumbnails of the WSI with the patch borders drawn in red.

<img src="_readme/viz.png" alt="Patches drawn on top of the original WSI." height="150px">

## Need for Speed
Trident offers two optional ways to meet those conference deadlines on short notice: caching and multiprocessing.

### Caching
If your WSIs are on a cloud directory, it may be beneficial to copy them to a local directory before feature extraction. This is because the time it takes to read a WSI from a remote location is often longer than the time it takes to process the WSI locally. To do this, you can specify a path for `wsi_cache` when initializing Trident (see `run_trident.py`). If you do not specify `wsi_cache`, Trident will process the WSIs directly from `wsi_source_dir`. Caching is only recommended if you plan to do feature extraction; otherwise, the benefit of caching is typically outweighed by the I/O cost of copying the WSIs.

Here is an example workflow using caching:
1. Run segmentation and patching normally, without caching.
2. We will use the cache for feature extraction. To copy WSIs to the cache, run this command:
```bash
python run_batch_of_slides.py --task cache --job_dir ./trident_processed --wsi_dir wsis --wsi_cache cache_dir
```
3. While the WSIs are being transferred, you can start a feature extraction job in a separate terminal window, pointing to the same `wsi_cache` directory. For example:
```bash
python run_batch_of_slides.py --task feat --wsi_dir wsis --job_dir ./trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256 --wsi_cache cache_dir
```
This instance will automatically use the cached WSIs if they are available, or skip to the next WSI if they are not.

If you are running low on storage, you can set `clear_cache` to `True` in the feature extraction job. This will delete the cached WSIs as they are processed. Note that this assumes the caching job can run faster than the feature extraction job. Otherwise, the feature extraction job will skip slides because the caching job has not yet finished transferring them. If you find this is happening, you can rerun the feature extraction job once the caching job has finished.

> [!WARNING]
> Be careful when setting `clear_cache` to `True`. Make sure `wsi_cache` is set to the correct directory. Otherwise, you may accidentally delete your original copy of the raw WSIs.

### Multiprocessing
Trident supports flexible multiprocessing, so you can run many instances of caching, segmentation, patching, or feature extraction in parallel and they will automatically avoid conflicts by "leapfrogging" each other. Before processing a slide, Trident creates a lockfile of the form `{slide_name}.lock`. If another Trident instance tries to process the same slide, it will see the lock and skip to the next slide.

Here is an example workflow using multiprocessing:
1. Open a terminal window (it is highly recommended you use [tmux](https://github.com/tmux/tmux/wiki) so that processes continue running even if your computer sleeps) and start a segmentation job:
```bash
python run_batch_of_slides.py --task seg --wsi_dir ./wsis --job_dir ./trident_processed --gpu 0
```
2. Open another terminal window (or another tmux pane) and run the exact same command.
3. Repeat Step 2 one or more times to spawn additional processes depending on how powerful your computer is. Be careful not to overload your machine, see Tips below:

#### Tips:
- Running multiple instances in parallel does not necessarily speed up processing, because of CPU or I/O bottlenecks. For example, on my machine I find that running more than 3 feature extraction instances in parallel causes things to lock up. Running multiple instances of the same task is probably only helpful if your bottleneck is the GPU. Keep track of your CPU load using `htop`.
- If you are concurrently running two consecutive tasks (e.g., patching and feature extraction), make sure that the upstream task is faster than the downstream task. Otherwise, the downstream task will end up skipping slides because the upstream task has not yet finished processing them. In case this happens, just rerun the downstream task once the upstream task has finished.

## Custom Pipelines

Trident provides a simple `encoder_factory` function for loading many patch and slide encoders through a unified API. You can therefore load encoders into your own pipeline for inference or finetuning.

### Patch Encoders
```python
from trident.patch_encoder_models.load import encoder_factory
encoder = encoder_factory("uni_v1") # Or any other encoder name
# also comes with:
print(encoder.enc_name)         # Model name 
print(encoder.eval_transforms)  # PyTorch transforms to process the input image
print(encoder.precision)        # Recommended precision to run the model
```

### Slide Encoders
```python
from trident.slide_encoder_models.load import encoder_factory
encoder = encoder_factory("titan") # Or any other encoder name
# also comes with:
print(encoder.enc_name)         # Model name
print(encoder.precision)        # Recommended precision to run the model
```

Some encoders take optional keyword arguments. For example, the `conch_v1` encoder can be run with or without the projection head. These keyword arguments can be passed directly to encoder_factory:

```python
encoder = encoder_factory("conch_v1", with_proj=True)
```

## Details

### Segmentation
Segmentation is performed by a [DeepLabV3 model](https://arxiv.org/abs/1706.05587v3) finetuned specifically to segment tissue from background in WSIs. By default, `run_trident.py` segments at 10x magnification. Use `--fast_seg` to segment at 5x magnification, which is faster but may be less accurate. If neither option yields good results, you can manually set `seg_mag` in `processor.run_segmentation_job()` to a higher magnification level. Note that higher magnification levels will be substantially slower.

### Patching
Trident's patching module is deterministic and extracts a set of patch coordinates given a tissue mask. Patches are extracted at a particular size (`patch_size`) and particular magnification (`mag`). Trident will attempt to read the base resolution of the WSI and calculate the appropriate downsample factor to extract patches at the desired magnification. Patches can either be nonoverlapping (`overlap == 0`) or overlapping (`overlap > 0`), where `overlap` refers to the absolute overlap in pixels. Trident keeps all patches that contain at least one pixel of tissue.

### Feature Extraction
Given a set of WSIs and patch coordinates, Trident's feature extraction module extracts features from the patches using a pretrained image encoder and associated transforms. Internally, Trident extracts patches from each WSI in batches and applies the provided transformations to them, then feeds them through the image encoder. The resulting patch features (shape (`num_patches`, `feature_dim`)) can be saved as h5 or pt files.

The `batch_size` parameter can be used to limit the number of patches processed in parallel. It is recommended to set `batch_size` as high as possible without running out of VRAM. The provided image encoder has only the following requirements: (1) must subclass `nn.Module`, (2) must have a `forward` method which takes a tensor of shape (b c h w) and returns a tensor of shape (b f). The image encoder must also have a `transforms` attribute, which should resize the image to the size expected by the image encoder and apply any other necessary transformations (including normalization and conversion to tensor). 

## 🙋 FAQ
- **Q**: How do I extract patch embeddings from legacy patch coordinates extracted with CLAM?
   - **A**:
      ```bash
      python run_batch_of_slides.py --task feat --wsi_dir ..wsis --job_dir legacy_dir --patch_encoder uni_v1 --mag 20 --patch_size 256 --coords_dir extracted_mag20x_patch256_fp/
      ```
- **Q**: How do I keep patches corresponding to holes in the tissue?
   - **A**: Set `holes_are_tissue` to `True` when running the segmentation job. Holes will be counted as part of tissue in contours and masks.

- **Q**: I see weird messages when building models using timm. What is happening?
   - **A**: Make sure your `timm==0.9.16` is installed. `timm==1.X.X` creates issues with most models. 

- **Q**: How can I use `run_single_slide.py` and `run_batch_of_slides.py` in other repos with minimal work?
  - **A**: Make sure `trident` is installed using `pip install -e .`. Then, both scripts are exposed and can be integrated into any Python code, e.g., as

```python
import sys 
from run_single_slide import main

sys.argv = [
    "run_single_slide",
    '--slide_path', "output/wsis/394140.svs",
    "--job_dir", 'output/',
    "--mag", "20",
    "--patch_size", '256'
]

main()
```

- **Q**: I am not satisfied with the tissue vs background segmentation. What can I do?
   - **A**: Trident uses GeoJSON to store and load segmentations. This format is also compatible with [QuPath](https://qupath.github.io/), enabling seamless integration. You can load the Trident segmentation into QuPath, modify it using QuPath's annotation tools, and save the updated segmentation back to GeoJSON. You can also force the segmentation to happen at 20x (which will be slower). 

- **Q**: I want to process a custom list of WSIs. Can I do it? Also, most of my WSIs don't have the micron per pixel (mpp) stored. Can I pass it?
   - **A**: Yes using the `--custom_list_of_wsis` argument. Provide a list of WSI names in a CSV (with slide extension, `wsi`). Optionally, provide the mpp (field `mpp`)
 
 - **Q**: Do I need to install any additional packages to use Trident?
   - **A**: Most models require additional installation (e.g., CtransPath feature extraction require to run `pip install timm_ctp`). Follow error messages that provide the steps to run your favorite model. 

## License and Terms of Tuse

ⓒ Mahmood Lab. This repository is released under the [CC-BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en) license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of this repository is prohibited and requires prior approval. By downloading patch encoders, you agree to follow the model's respective license.

## Acknowledgements

The project was built on top of amazing repositories such as [Timm](https://github.com/huggingface/pytorch-image-models/), [HuggingFace](https://huggingface.co/docs/datasets/en/index), and open-source contributions from the community. We thank the authors and developers for their contribution. 

## Issues

- The preferred mode of communication is via GitHub issues.
- If GitHub issues are inappropriate, email gjaume@bwh.harvard.edu (and cc andrewzh@mit.edu).
- Immediate response to minor issues may not be available.

## Reference

If you find our work useful in your research or if you use parts of this code, please consider citing our paper:

```
@article{vaidya2025molecular,
  title={Molecular-driven Foundation Model for Oncologic Pathology},
  author={Vaidya, Anurag and Zhang, Andrew and Jaume, Guillaume and Song, Andrew H and Ding, Tong and Wagner, Sophia J and Lu, Ming Y and Doucet, Paul and Robertson, Harry and Almagro-Perez, Cristina and others},
  journal={arXiv preprint arXiv:2501.16652},
  year={2025}
}

@article{zhang2025standardizing,
  title={Standardizing Preprocessing and Benchmarking of AI Models for Pathology},
  author={Zhang, Andrew and Jaume, Guillaume and Vaidya, Anurag and others},
  journal={},
  year={2025}
}
```

<img src=.github/joint_logo.png> 
