"""Setup script for mlp package."""

from setuptools import setup

setup(
    name="mlp",
    author="Pawel Swietojanski, Steve Renals, Matt Graham and Antreas Antoniou",
    description=(
        "Neural network framework for University of Edinburgh "
        "School of Informatics Machine Learning Practical course."
    ),
    url="https://github.com/CSTR-Edinburgh/mlpractical",
    packages=["mlp"],
    install_requires=["tqdm>=4.60,<5"],

)
