from setuptools import setup, find_packages

setup(
    name='trident',
    version='0.0.2',
    packages=find_packages(),
    install_requires=[
        'ipywidgets',
        'torch',
        'transformers',
        'tqdm',
        'h5py',
        'matplotlib',
        'segmentation-models-pytorch',
        'opencv-python',
        'openslide-python',
        'Pillow',
        'timm==0.9.16',
        'einops_exts',
        'geopandas',
        'huggingface_hub',
        'openslide-bin',
        'scipy'
    ],
    entry_points={
        'console_scripts': [
            'run_batch_of_slides=run_batch_of_slides:main',  
            'run_single_slide=run_single_slide:main',  
        ],
    },
    author="Andrew Zhang, Guillaume Jaume, Paul Doucet",
    author_email="andrewzh@mit.edu, gjaume@bwh.harvard.edu, homedoucetpaul@gmail.com",
    description="A package for preprocessing whole-slide images.",
    url="https://github.com/mahmoodlab/trident",
    package_data={
        'trident.slide_encoder_models': ['local_ckpts.json'],
        'trident.patch_encoder_models': ['local_ckpts.json'],
    },
    include_package_data=True,  # Required for package_data to work
)
