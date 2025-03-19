from setuptools import setup, find_packages

setup(
    name="ncaa_pytorch_predictor",
    version="0.1.0",
    description="NCAA Tournament Predictor using PyTorch",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "shap>=0.41.0",
    ],
    python_requires=">=3.8",
)