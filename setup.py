from pathlib import Path
from setuptools import find_packages, setup

from transformers_finetuner import __version__

setup(
    name="transformers_finetuner",
    version=__version__,
    description="A finetuner utility for transformers models",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    keywords="transformers fine-tuning machine-learning deep-learning",
    packages=find_packages(include=["transformers_finetuner", "transformers_finetuner.*"]),
    url="https://github.com/BramVanroy/transformers-finetuner/",
    author="Bram Vanroy",
    author_email="bramvanroy@hotmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Text Processing",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    project_urls={
        "Issue tracker": "https://github.com/BramVanroy/transformers-finetuner/issues",
        "Source": "https://github.com/BramVanroy/transformers-finetuner"
    },
    python_requires=">=3.7",
    install_requires=[
        "datasets",
        "GitPython",
        "matplotlib",
        "pandas",
        "ray[tune]",
        "seaborn",
        "scikit-learn",
        "torch",
        "transformers>=4.21.0"
    ],
    entry_points={
        "console_scripts": ["finetune=transformers_finetuner.train:main"]
    }
)
