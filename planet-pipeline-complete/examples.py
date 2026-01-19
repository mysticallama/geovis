"""
Example Usage Scripts for Planet Labs Imagery Processing Pipeline

This file contains several example workflows demonstrating different use cases.
"""

from planet_pipeline import PlanetPipeline
from pathlib import Path


def example_1_basic_workflow():
    """
    Example 1: Basic workflow with single AOI
    Query, download, and calculate indices for one area.
    """
    print("=" * 60)
    print("Example 1: Basic Workflow")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = PlanetPipeline(storage_dir="./data/example1")
    
    # Add single AOI
    pipeline.add_aoi(
        name="test_site",
        geometry={
            "type": "Polygon",
            "coordinates": [[
                [-122.5, 37.7],
                [-122.5, 37.8],
                [-122.4, 37.8],
                [-122.4, 37.7],
                [-122.5, 37.7]
            ]]
        },
        metadata={"description": "Test site in San Francisco Bay Area"}
    )
    
    # Query imagery
    print("\nQuerying Planet API...")
    results = pipeline.query_all_aois(
        start_date="2024-01-01",
        end_date="2024-01-31",
        cloud_cover_max=0.1,
        max_gsd=3.5
    )
    
    print(f"Found {len(results['test_site'])} scenes")
    
    # Download (limit to 5 scenes for example)
    print("\nDownloading imagery...")
    downloaded = pipeline.download_imagery(
        asset_types=["ortho_analytic_4b"],
        limit_per_aoi=5
    )
    
    # Calculate indices
    print("\nCalculating spectral indices...")
    indices_results = pipeline.calculate_indices(
        indices=["ndvi", "ndwi", "evi"]
    )
    
    # Export metadata
    pipeline.export_metadata("./data/example1/metadata.json")
    
    print("\n✓ Example 1 complete!")
    print(f"Data saved to: ./data/example1")


def example_2_multi_aoi():
    """
    Example 2: Multiple AOIs from file
    Process multiple agricultural fields simultaneously.
    """
    print("\n" + "=" * 60)
    print("Example 2: Multi-AOI Processing")
    print("=" * 60)
    
    pipeline = PlanetPipeline(storage_dir="./data/example2")
    
    # Create example AOI file
    import json
    aois_data = {
        "field_1": {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-121.5, 37.5],
                    [-121.5, 37.51],
                    [-121.49, 37.51],
                    [-121.49, 37.5],
                    [-121.5, 37.5]
                ]]
            },
            "metadata": {"crop": "corn", "field_id": "F001"}
        },
        "field_2": {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-121.6, 37.6],
                    [-121.6, 37.61],
                    [-121.59, 37.61],
                    [-121.59, 37.6],
                    [-121.6, 37.6]
                ]]
            },
            "metadata": {"crop": "wheat", "field_id": "F002"}
        }
    }
    
    # Save AOI file
    aoi_file = Path("./data/example2/aois.json")
    aoi_file.parent.mkdir(parents=True, exist_ok=True)
    with open(aoi_file, 'w') as f:
        json.dump(aois_data, f, indent=2)
    
    # Load AOIs from file
    pipeline.add_aois_from_file(str(aoi_file))
    
    # Query all AOIs
    print("\nQuerying multiple AOIs...")
    results = pipeline.query_all_aois(
        start_date="2024-06-01",
        end_date="2024-08-31",
        cloud_cover_max=0.05
    )
    
    for aoi_name, scenes in results.items():
        print(f"{aoi_name}: {len(scenes)} scenes")
    
    # Download and process
    print("\nDownloading imagery...")
    pipeline.download_imagery(limit_per_aoi=10)
    
    print("\nCalculating vegetation indices...")
    pipeline.calculate_indices(indices=["ndvi", "evi", "savi"])
    
    # Get summary
    summary = pipeline.get_summary()
    print(f"\nPipeline Summary:")
    print(f"  Total AOIs: {summary['num_aois']}")
    print(f"  Total scenes: {sum(summary['search_results'].values())}")
    
    print("\n✓ Example 2 complete!")


def example_3_ml_dataset():
    """
    Example 3: Prepare ML training dataset
    Create PyTorch-ready dataset with data augmentation.
    """
    print("\n" + "=" * 60)
    print("Example 3: ML Dataset Preparation")
    print("=" * 60)
    
    pipeline = PlanetPipeline(storage_dir="./data/example3")
    
    # Add labeled AOIs for different classes
    classes = {
        "forest": [-122.0, 37.9],
        "urban": [-122.4, 37.8],
        "agriculture": [-121.8, 37.5]
    }
    
    for class_name, coords in classes.items():
        pipeline.add_aoi(
            name=class_name,
            geometry={
                "type": "Point",
                "coordinates": coords
            },
            metadata={"label": class_name}
        )
    
    # Query imagery
    print("\nQuerying imagery for all classes...")
    pipeline.query_all_aois(
        start_date="2024-01-01",
        end_date="2024-12-31",
        cloud_cover_max=0.1
    )
    
    # Download
    print("\nDownloading imagery...")
    pipeline.download_imagery(limit_per_aoi=20)
    
    # Prepare ML dataset
    print("\nPreparing PyTorch dataset...")
    dataset_path = pipeline.prepare_for_ml(
        model_type="pytorch",
        output_format="chips",
        chip_size=256,
        overlap=32,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        normalize=True,
        augment=True
    )
    
    print(f"\n✓ Dataset ready at: {dataset_path}")
    print("  Use pytorch_loader.py to load the dataset")


def example_4_custom_processing():
    """
    Example 4: Custom preprocessing pipeline
    Apply specific preprocessing operations.
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Processing Pipeline")
    print("=" * 60)
    
    from planet_pipeline.preprocessing import ImagePreprocessor
    from planet_pipeline.storage import ImageryStorage
    
    storage = ImageryStorage(base_dir="./data/example4")
    preprocessor = ImagePreprocessor()
    
    # For this example, we'll simulate having some imagery
    print("\nApplying custom preprocessing operations...")
    print("Operations: cloud_mask → atmospheric_correction → normalization")
    
    # Example of how you would process imagery
    """
    imagery_files = storage.get_imagery_files("my_aoi")
    
    for img_file in imagery_files:
        processed = preprocessor.process(
            input_path=img_file,
            operations=[
                "cloud_mask",
                "atmospheric_correction",
                "normalization"
            ],
            output_dir=storage.get_processed_dir("my_aoi"),
            cloud_threshold=0.3,
            method="percentile"
        )
        print(f"Processed: {processed}")
    """
    
    print("\n✓ Example 4 demonstrates custom processing")


def example_5_indices_batch():
    """
    Example 5: Batch calculate all indices
    Calculate all available spectral indices.
    """
    print("\n" + "=" * 60)
    print("Example 5: Batch Indices Calculation")
    print("=" * 60)
    
    from planet_pipeline.indices import SpectralIndices
    
    indices_calc = SpectralIndices()
    
    # Show all available indices
    available = indices_calc.list_available_indices()
    print(f"\nAvailable indices ({len(available)}):")
    for idx in available:
        print(f"  - {idx}")
    
    print("\n✓ Example 5 shows available indices")


def main():
    """Run all examples."""
    import sys
    
    print("\n" + "=" * 60)
    print("Planet Labs Imagery Pipeline - Example Scripts")
    print("=" * 60)
    
    examples = {
        "1": ("Basic Workflow", example_1_basic_workflow),
        "2": ("Multi-AOI Processing", example_2_multi_aoi),
        "3": ("ML Dataset Preparation", example_3_ml_dataset),
        "4": ("Custom Processing", example_4_custom_processing),
        "5": ("Batch Indices", example_5_indices_batch)
    }
    
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  all. Run all examples")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nSelect example (1-5, all, or q to quit): ").strip()
    
    if choice.lower() == 'q':
        print("Exiting...")
        return
    
    if choice.lower() == 'all':
        for _, func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
    elif choice in examples:
        try:
            examples[choice][1]()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice")
    
    print("\n" + "=" * 60)
    print("Examples complete! Check ./data/ for outputs")
    print("=" * 60)


if __name__ == "__main__":
    # Note: These examples require a valid Planet API key
    # Set PL_API_KEY environment variable before running
    
    import os
    if not os.environ.get("PL_API_KEY"):
        print("⚠️  Warning: PL_API_KEY not set")
        print("Set environment variable: export PL_API_KEY='your_key'")
        print("\nRunning examples in demo mode (will show structure only)")
        print()
    
    main()
