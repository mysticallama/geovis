"""
Setup script for Planet Labs Imagery Processing Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="planet-pipeline",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Modular pipeline for Planet Labs satellite imagery processing and ML preparation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/planet-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "numpy>=1.24.0",
        "rasterio>=1.3.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
        "shapely>=2.0.0",
        "pyproj>=3.5.0",
    ],
    extras_require={
        "ml": [
            "torch>=2.0.0",
            "tensorflow>=2.12.0",
            "scikit-learn>=1.3.0",
        ],
        "geo": [
            "geopandas>=0.13.0",
        ],
        "osm": [
            # OSM Overpass module - core deps already included
            # Optional: PIL/Pillow for mask visualization
            "Pillow>=9.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "all": [
            "torch>=2.0.0",
            "tensorflow>=2.12.0",
            "scikit-learn>=1.3.0",
            "geopandas>=0.13.0",
            "Pillow>=9.0.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "planet-pipeline=planet_pipeline:main",
        ],
    },
    include_package_data=True,
    keywords="planet satellite imagery remote-sensing machine-learning deep-learning pytorch tensorflow osm openstreetmap overpass geojson yolo siamese dbscan",
)
