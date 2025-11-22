"""
Environment Setup Script
Creates all necessary directories for the project
"""

import os
from pathlib import Path

def create_directory_structure():
    """Create the complete project directory structure"""
    
    base_dir = Path(__file__).parent
    
    directories = [
        # Data directories
        base_dir / "data" / "raw",
        base_dir / "data" / "processed",
        base_dir / "data" / "features",
        
        # Source code directories
        base_dir / "src" / "data_processing",
        base_dir / "src" / "models",
        base_dir / "src" / "visualization",
        base_dir / "src" / "api",
        
        # Notebook directories
        base_dir / "notebooks" / "exploration",
        base_dir / "notebooks" / "experiments",
        
        # Other directories
        base_dir / "config",
        base_dir / "tests",
        base_dir / "logs",
    ]
    
    created = []
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        created.append(str(directory))
        print(f"✓ Created: {directory}")
    
    # Create __init__.py files if they don't exist
    init_files = [
        base_dir / "notebooks" / "__init__.py",
        base_dir / "notebooks" / "exploration" / "__init__.py",
        base_dir / "notebooks" / "experiments" / "__init__.py",
    ]
    
    for init_file in init_files:
        if not init_file.exists():
            init_file.write_text('"""Notebooks directory"""\n')
            print(f"✓ Created: {init_file}")
    
    print(f"\n✅ Environment setup complete! Created {len(created)} directories.")
    return created

if __name__ == "__main__":
    create_directory_structure()

