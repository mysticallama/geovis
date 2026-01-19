#!/usr/bin/env python3
"""
Setup Script for Planet Pipeline
This script will organize your downloaded files into the correct directory structure.

Usage:
    1. Download all files to a single directory
    2. Run: python SETUP_INSTRUCTIONS.py
    3. Files will be organized into planet_pipeline/ folder
"""

import os
import shutil
from pathlib import Path


def setup_pipeline_structure():
    """Create the proper directory structure and move files."""
    
    print("=" * 60)
    print("Planet Pipeline Setup")
    print("=" * 60)
    
    # Current directory
    current_dir = Path.cwd()
    
    # Create planet_pipeline directory
    pipeline_dir = current_dir / "planet_pipeline"
    pipeline_dir.mkdir(exist_ok=True)
    print(f"\n✓ Created directory: {pipeline_dir}")
    
    # Files to move into planet_pipeline/
    module_files = {
        "planet_pipeline_main.py": "__init__.py",
        "planet_pipeline_query.py": "query.py",
        "planet_pipeline_storage.py": "storage.py",
        "planet_pipeline_download.py": "download.py",
        "planet_pipeline_indices.py": "indices.py",
        "planet_pipeline_preprocessing.py": "preprocessing.py",
        "planet_pipeline_ml_prep.py": "ml_prep.py"
    }
    
    # Move and rename module files
    print("\nMoving module files...")
    for source_name, dest_name in module_files.items():
        source = current_dir / source_name
        dest = pipeline_dir / dest_name
        
        if source.exists():
            shutil.move(str(source), str(dest))
            print(f"  ✓ {source_name} → planet_pipeline/{dest_name}")
        else:
            print(f"  ⚠ {source_name} not found - please download it")
    
    # Check for other important files
    important_files = [
        "requirements.txt",
        "README.md",
        "setup.py",
        "examples.py",
        ".gitignore",
        ".env.example",
        "config.yaml.template",
        "example_aoi.geojson"
    ]
    
    print("\nChecking for important files...")
    for filename in important_files:
        if (current_dir / filename).exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ⚠ {filename} - please download it")
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nYour directory structure should now look like:")
    print("""
    your-project/
    ├── planet_pipeline/
    │   ├── __init__.py
    │   ├── query.py
    │   ├── storage.py
    │   ├── download.py
    │   ├── indices.py
    │   ├── preprocessing.py
    │   └── ml_prep.py
    ├── examples.py
    ├── setup.py
    ├── requirements.txt
    ├── README.md
    └── ...
    """)
    
    print("\nNext steps:")
    print("1. Create virtual environment: python -m venv venv")
    print("2. Activate it: source venv/bin/activate (Mac/Linux) or venv\\Scripts\\activate (Windows)")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Set up .env file with your Planet API key")
    print("5. Try examples: python examples.py")
    print("\n")


if __name__ == "__main__":
    try:
        setup_pipeline_structure()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
