"""
Run data validation and generate initial EDA
This script combines validation and initial exploration
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.data_loader import DataLoader, validate_data_completeness, create_data_quality_report

def main():
    """Run validation and create summary"""
    print("=" * 80)
    print("DATA VALIDATION & INITIAL EDA")
    print("=" * 80)
    print()
    
    # Initialize loader
    print("Step 1: Initializing data loader...")
    loader = DataLoader()
    print(f"✓ Loader initialized with {len(loader.venues)} venues")
    print()
    
    # Run validation
    print("Step 2: Running data validation...")
    print("This may take a few minutes...")
    print()
    
    validation_df = validate_data_completeness(loader)
    
    # Generate report
    print("Step 3: Generating data quality report...")
    report = create_data_quality_report(loader, output_path="data_quality_report.txt")
    print("✓ Report saved to data_quality_report.txt")
    print()
    
    # Quick summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total file checks: {len(validation_df)}")
    print(f"Files found: {validation_df['found'].sum()}")
    print(f"Files missing: {(~validation_df['found']).sum()}")
    print(f"Overall completeness: {validation_df['found'].mean() * 100:.1f}%")
    print()
    
    # Per venue summary
    print("Per Venue Breakdown:")
    print("-" * 80)
    for venue in loader.venues:
        venue_data = validation_df[validation_df['venue'] == venue]
        found = venue_data['found'].sum()
        total = len(venue_data)
        rows = venue_data['rows'].sum()
        print(f"{venue:30s}: {found:2d}/{total:2d} files ({found/total*100:5.1f}%) - {rows:6,} rows")
    print()
    
    # Test loading a sample file
    print("Step 4: Testing data loading...")
    print("-" * 80)
    
    # Test results
    results = loader.load_results_file("barber", "Race 1", "provisional")
    if not results.empty:
        print(f"✓ Results: {len(results)} rows loaded")
        print(f"  Columns: {len(results.columns)}")
        if 'NUMBER' in results.columns:
            print(f"  Drivers: {results['NUMBER'].nunique()}")
    else:
        print("✗ Results: No data found")
    
    # Test weather
    weather = loader.load_weather_data("barber", "Race 1")
    if not weather.empty:
        print(f"✓ Weather: {len(weather)} rows loaded")
    else:
        print("✗ Weather: No data found")
    
    # Test best laps
    best_laps = loader.load_best_laps("barber", "Race 1")
    if not best_laps.empty:
        print(f"✓ Best Laps: {len(best_laps)} rows loaded")
    else:
        print("✗ Best Laps: No data found")
    
    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Review data_quality_report.txt for detailed analysis")
    print("2. Open notebooks/exploration/01_initial_eda.ipynb for interactive EDA")
    print("3. Proceed to Phase 1: Track DNA Extraction")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()

