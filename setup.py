"""Setup configuration for the vietnerm package."""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="vietnerm",
    version="0.2.3",
    description=(
        "PhoBERT NER pipeline for Vietnamese document entity extraction"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="VietNerm Team",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(exclude=["raw_code", "raw_code.*", "docs"]),
    include_package_data=True,
    package_data={
        "": [
            "registry/*.yaml",
            "templates/**/*.yaml",
            "templates/**/*.txt",
            "training/config/*.yaml",
        ],
    },
    install_requires=[
        "torch>=1.13.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "seqeval>=1.2.2",
        "numpy>=1.24.0",
        "huggingface_hub>=0.16.0",
        "pyyaml>=6.0",
        "Jinja2>=3.1.0",
    ],
    entry_points={
        "console_scripts": [
            "vietnerm-generate=synthetic.generate_dataset:main",
            "vietnerm-train=training.train:main",
        ],
    },
)
