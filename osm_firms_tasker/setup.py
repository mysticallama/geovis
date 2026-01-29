"""Setup script for water_pipeline package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text()

setup(
    name="water-pipeline",
    version="1.0.0",
    description="Water infrastructure monitoring using satellite data, OSM, FIRMS, and ACLED",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/water-pipeline",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "shapely>=2.0.0",
        "geopandas>=0.14.0",
        "pyproj>=3.5.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "ml": [
            "torch",
            "scikit-learn",
        ],
        "viz": [
            "Pillow",
            "matplotlib",
        ],
        "imagery": [
            "rasterio>=1.3.0",
        ],
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "water-pipeline=water_pipeline.pipeline:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    keywords="gis satellite imagery water infrastructure monitoring osm firms acled",
)
