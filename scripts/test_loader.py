"""
Quick test script for the data loader
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.data_loader import DataLoader

def test_loader():
    """Test the data loader with actual data"""
    print("=" * 80)
    print("TESTING DATA LOADER")
    print("=" * 80)
    print()
    
    # Initialize loader
    loader = DataLoader()
    print(f"✓ Loader initialized")
    print(f"  Venues: {', '.join(loader.venues)}")
    print()
    
    # Test 1: Load results file
    print("Test 1: Loading race results...")
    try:
        results = loader.load_results_file("barber", "Race 1", "provisional")
        if not results.empty:
            print(f"  ✓ Success! Loaded {len(results)} rows")
            print(f"    Columns: {len(results.columns)}")
            print(f"    Sample columns: {list(results.columns[:5])}")
        else:
            print("  ⚠ No data found")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()
    
    # Test 2: Load weather data
    print("Test 2: Loading weather data...")
    try:
        weather = loader.load_weather_data("barber", "Race 1")
        if not weather.empty:
            print(f"  ✓ Success! Loaded {len(weather)} rows")
            print(f"    Columns: {list(weather.columns)}")
        else:
            print("  ⚠ No data found")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()
    
    # Test 3: Load best laps
    print("Test 3: Loading best laps...")
    try:
        best_laps = loader.load_best_laps("barber", "Race 1")
        if not best_laps.empty:
            print(f"  ✓ Success! Loaded {len(best_laps)} rows")
            print(f"    Drivers: {best_laps['NUMBER'].nunique() if 'NUMBER' in best_laps.columns else 'N/A'}")
        else:
            print("  ⚠ No data found")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()
    
    # Test 4: Load analysis
    print("Test 4: Loading analysis data...")
    try:
        analysis = loader.load_analysis_endurance("barber", "Race 1")
        if not analysis.empty:
            print(f"  ✓ Success! Loaded {len(analysis)} rows")
            print(f"    Unique drivers: {analysis['NUMBER'].nunique() if 'NUMBER' in analysis.columns else 'N/A'}")
        else:
            print("  ⚠ No data found")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()
    
    # Test 5: Load lap timing
    print("Test 5: Loading lap timing...")
    try:
        lap_time = loader.load_lap_timing("barber", "Race 1", "time")
        if not lap_time.empty:
            print(f"  ✓ Success! Loaded {len(lap_time)} rows")
        else:
            print("  ⚠ No data found")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()
    
    # Test 6: Load telemetry sample
    print("Test 6: Loading telemetry sample (first 1000 rows)...")
    try:
        telemetry = loader.load_telemetry("barber", "Race 1", sample_size=1000)
        if not telemetry.empty:
            print(f"  ✓ Success! Loaded {len(telemetry)} rows")
            print(f"    Unique parameters: {telemetry['parameter_name'].nunique() if 'parameter_name' in telemetry.columns else 'N/A'}")
        else:
            print("  ⚠ No data found")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()
    
    # Test 7: Load championship data
    print("Test 7: Loading championship data...")
    try:
        championship = loader.load_championship_data()
        if not championship.empty:
            print(f"  ✓ Success! Loaded {len(championship)} rows")
            print(f"    Drivers: {len(championship)}")
        else:
            print("  ⚠ No data found")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()
    
    print("=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_loader()

