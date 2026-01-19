# Planet Labs Imagery Processing Pipeline

A comprehensive, modular Python pipeline for querying, downloading, processing, and preparing Planet Labs satellite imagery for machine learning workflows.

## 🚀 Features

### Multi-AOI Management
- Query multiple Areas of Interest (AOIs) simultaneously
- Support for GeoJSON, individual geometries, and bulk AOI loading
- Organized storage with automatic metadata tracking

### Imagery Processing
- **Download Management**: Parallel downloads with progress tracking and resume capability
- **Spectral Indices**: Calculate 14+ indices (NDVI, NDWI, EVI, SAVI, etc.)
- **Preprocessing**: Cloud masking, atmospheric correction, normalization
- **ML Preparation**: Generate training datasets for PyTorch, TensorFlow, and scikit-learn

### Flexible Architecture
- Modular design for easy integration into existing pipelines
- Can be used as standalone tool or imported as Python library
- Extensible with custom processing operations

## 🔧 Installation

```bash
# Clone repository
git clone <your-repo-url>
cd planet-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
export PL_API_KEY='your-planet-api-key'
```

## 🚀 Quick Start

### As a Python Library

```python
from planet_pipeline import PlanetPipeline

# Initialize pipeline
pipeline = PlanetPipeline(storage_dir="./my_data")

# Add AOIs
pipeline.add_aoi("san_francisco", geometry_file="sf_aoi.geojson")

# Query imagery
pipeline.query_all_aois(
    start_date="2024-01-01",
    end_date="2024-01-31",
    cloud_cover_max=0.1
)

# Download imagery
pipeline.download_imagery(limit_per_aoi=10)

# Calculate spectral indices
pipeline.calculate_indices(indices=["ndvi", "ndwi", "evi"])

# Prepare for ML
dataset_path = pipeline.prepare_for_ml(
    model_type="pytorch",
    chip_size=256
)
```

## 📦 Pipeline Modules

1. **Query** (`query.py`) - Enhanced API client with retry and caching
2. **Storage** (`storage.py`) - Organized file system management
3. **Download** (`download.py`) - Parallel downloads with progress tracking
4. **Indices** (`indices.py`) - 14+ spectral indices calculation
5. **Preprocessing** (`preprocessing.py`) - Cloud masking, correction, normalization
6. **ML Prep** (`ml_prep.py`) - Dataset preparation for PyTorch/TensorFlow/sklearn

## 💡 Usage Examples

### Multi-AOI Batch Processing

```python
pipeline = PlanetPipeline(storage_dir="./agricultural_study")

# Load multiple AOIs
pipeline.add_aois_from_file("farm_fields.geojson")

# Query all fields
results = pipeline.query_all_aois(
    start_date="2024-06-01",
    end_date="2024-08-31",
    max_gsd=3.0,
    cloud_cover_max=0.05
)

# Download and process
pipeline.download_imagery(limit_per_aoi=20)
pipeline.calculate_indices(indices=["ndvi", "evi", "savi"])
```

### ML Training Dataset

```python
# Add labeled AOIs
pipeline.add_aoi("forest", geometry_file="forest_samples.geojson")
pipeline.add_aoi("urban", geometry_file="urban_samples.geojson")
pipeline.add_aoi("water", geometry_file="water_samples.geojson")

# Prepare dataset
dataset_path = pipeline.prepare_for_ml(
    model_type="pytorch",
    chip_size=256,
    overlap=32,
    train_split=0.7,
    augment=True,
    label_file="labels.json"
)
```

## 🔍 Available Spectral Indices

| Index | Use Case | Formula |
|-------|----------|---------|
| NDVI | Vegetation health | (NIR-Red)/(NIR+Red) |
| NDWI | Water bodies | (Green-NIR)/(Green+NIR) |
| EVI | Dense vegetation | 2.5×((NIR-Red)/(NIR+6×Red-7.5×Blue+1)) |
| SAVI | Sparse vegetation | ((NIR-Red)/(NIR+Red+0.5))×1.5 |
| GNDVI | Chlorophyll | (NIR-Green)/(NIR+Green) |
| NDBI | Urban areas | (SWIR-NIR)/(SWIR+NIR) |
| BAI | Burned areas | 1/((0.1-Red)²+(0.06-NIR)²) |

Plus: MSAVI, NBR, NDMI, ARVI, GCI, SIPI, VARI

## 🤖 ML Integration

### PyTorch

```python
from pytorch_loader import PlanetDataset
from torch.utils.data import DataLoader

train_dataset = PlanetDataset("dataset_path/train")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Your training loop here
for chips, labels in train_loader:
    # Train model
    pass
```

### TensorFlow

```python
from tensorflow_loader import create_dataset

train_dataset = create_dataset("dataset_path/train", batch_size=32)
model.fit(train_dataset, epochs=10)
```

## 📊 Directory Structure

```
planet_data/
├── aois/              # AOI definitions
├── imagery/           # Downloaded imagery by AOI/date
├── processed/         # Preprocessed imagery
├── indices/           # Spectral indices
├── ml_datasets/       # ML-ready datasets
└── cache/             # API response cache
```

## 🧪 Testing

```bash
pytest tests/
pytest --cov=planet_pipeline tests/
```

## 📝 License

MIT License

## 🙏 Acknowledgments

- Planet Labs for satellite imagery and API
- Open source geospatial community

## 📧 Support

- GitHub Issues
- [Planet API Documentation](https://developers.planet.com/docs/apis/)
