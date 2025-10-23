import argparse
import datetime
import sys
from datetime import date
from pathlib import Path

from setuptools import find_packages, setup

# update this version when a new official release is made
__version__ = "1.0.0"


def get_package_version():
    return __version__


def get_nightly_version():
    today = date.today()
    now = datetime.datetime.now()
    timing = f"{now.hour:02d}{now.minute:02d}"
    return f"{today.year}.{today.month}.{today.day}.{timing}"


def get_python_version():
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def get_dependencies():
    """
    Dependencies for NEBULA-Alpha.
    Based on the same core dependencies as ManiSkill, plus additional packages.
    """
    install_requires = [
        # Core dependencies (same as ManiSkill)
        "numpy>=1.22,<2.0.0",
        "scipy>=1.15.0",
        "dacite>=1.9.0",
        "gymnasium==0.29.1",
        "h5py>=3.15.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.67.0",
        "GitPython>=3.1.44",
        "tabulate>=0.9.0",
        "transforms3d>=0.4.2",
        "trimesh>=4.6.0",
        "imageio>=2.37.0",
        "imageio[ffmpeg]",
        "mplib==0.1.1;platform_system=='Linux'",
        "fast_kinematics==0.2.2;platform_system=='Linux'",
        "IPython>=8.0.0",
        "pytorch_kinematics==0.7.5",
        "pynvml>=12.0.0",
        "tyro>=0.9.35",
        "huggingface_hub>=0.35.0",
        "sapien>=3.0.0;platform_system=='Linux'",
        "sapien>=3.0.0.b1;platform_system=='Windows'",
        
        # Additional dependencies for NEBULA-Alpha
        "opencv-python>=4.11.0",
        "matplotlib>=3.10.0",
        "lxml>=5.4.0",
        "pandas>=2.2.0",
        "pillow>=12.0.0",
        "hjson>=3.1.0",
        "jsonlines>=4.0.0",
        "scikit-image>=0.25.0",
        "shapely>=2.1.0",
        "sentencepiece>=0.2.0",
        "safetensors>=0.5.0",
        "imgaug>=0.4.0",
        "arm_pytorch_utilities>=0.4.3",
        "toppra>=0.6.3",
        "pytorch-seed>=0.2.0",
        "psutil>=7.0.0",
        "setproctitle>=1.3.0",
        
        # Development and utilities
        "click>=8.2.0",
        "rich>=14.0.0",
        "cloudpickle>=3.1.0",
        "filelock>=3.13.0",
        "fsspec>=2024.6.0",
        "requests>=2.32.0",
        "certifi>=2025.0.0",
    ]
    # NOTE: For macOS users, install sapien manually from:
    # https://github.com/haosulab/SAPIEN/releases
    return install_requires


def parse_args(argv):
    parser = argparse.ArgumentParser(description="NEBULA-Alpha setup.py configuration")
    parser.add_argument(
        "--package_name",
        type=str,
        default="nebula-alpha",
        choices=["nebula-alpha", "nebula-alpha-nightly"],
        help="the name of this output wheel. Should be either 'nebula-alpha' or 'nebula-alpha-nightly'",
    )
    return parser.parse_known_args(argv)


def main(argv):

    args, unknown = parse_args(argv)
    name = args.package_name
    is_nightly = name == "nebula-alpha-nightly"

    this_directory = Path(__file__).parent
    long_description = (this_directory / "README.md").read_text(encoding="utf8")

    if is_nightly:
        version = get_nightly_version()
    else:
        version = get_package_version()

    sys.argv = [sys.argv[0]] + unknown
    print(sys.argv)
    setup(
        name=name,
        version=version,
        description="NEBULA-Alpha: An advanced robotics simulation environment with ManiSkill-compatible dependencies",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="NEBULA-Alpha contributors",
        url="https://github.com/yourusername/nebula-alpha",  # Update with your actual URL
        packages=find_packages(include=["nebula*"]),
        python_requires=">=3.9",
        setup_requires=["setuptools>=62.3.0"],
        install_requires=get_dependencies(),
        # Include all assets and data files
        package_data={
            "nebula": [
                "**/*.glb",
                "**/*.urdf",
                "**/*.xml",
                "**/*.json",
                "**/*.yaml",
                "**/*.yml",
                "assets/**",
                "configs/**",
            ],
        },
        extras_require={
            "dev": [
                "pytest>=8.0.0",
                "black",
                "isort",
                "pre-commit",
                "build",
                "twine",
                "pytest-xdist[psutil]",
                "pytest-forked",
            ],
            "models": [
                # Optional: torch dependencies (large download)
                # Users can install torch separately for their specific CUDA version
                "torch>=2.6.0",
                "torchvision>=0.21.0",
                "torchaudio>=2.6.0",
            ],
            "wandb": [
                # Optional: Weights & Biases for experiment tracking
                "wandb>=0.12.0",
                "docker-pycreds>=0.4.0",
                "sentry-sdk>=2.0.0",
            ],
        },
    )


if __name__ == "__main__":
    main(sys.argv[1:])

