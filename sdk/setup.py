"""Setup script for VietNerm SDK."""

from pathlib import Path

from setuptools import find_packages, setup

readme_path = Path(__file__).parent.parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="vietnerm",
    version="0.2.4",
    description="Vietnamese Document NER Extraction SDK using PhoBERT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="VietNerm Team",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.10",
        "transformers>=4.20",
        "numpy",
        "pyyaml",
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
