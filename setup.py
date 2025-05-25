from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ising_model",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python implementation of the 2D Ising model using the Metropolis algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ising_model",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.12b0",
            "isort>=5.10.1",
            "mypy>=0.930",
        ],
    },
    entry_points={
        "console_scripts": [
            "ising-sim=ising_model.cli:main",
        ],
    },
)
