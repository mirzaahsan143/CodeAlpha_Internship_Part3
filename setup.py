"""
Setup script for Leaf Disease Detection System.
"""
from setuptools import setup, find_packages

setup(
    name="leaf_disease_detection",
    version="1.0.0",
    description="Leaf Disease Detection System using Deep Learning",
    author="CodeAlpha Intern",
    author_email="intern@codealpha.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
        "Pillow>=10.0.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "leaf-disease-train=train:main",
            "leaf-disease-predict=predict:main",
            "leaf-disease-run=main:main",
        ],
    },
)
