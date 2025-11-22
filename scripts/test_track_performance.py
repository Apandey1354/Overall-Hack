"""
Test script for Track Performance Analyzer
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.track_performance_analyzer import TrackPerformanceAnalyzer, analyze_track_performance
from src.data_processing.data_loader import DataLoader

def test_performance_analyzer():
    """Test the Track Performance Analyzer"""
    print("=" * 80)
    print("TESTING TRACK PERFORMANCE ANALYZER")
    print("=" * 80)
    print()
    
    # Initialize
    loader = DataLoader()
    analyzer = TrackPerformanceAnalyzer(data_loader=loader)
    
    print(f"Venues available: {', '.join(loader.venues)}")
    print()
    
    # Test 1: Driver Performance by Track Type
    print("=" * 80)
    print("Test 1: Driver Performance by Track Type")
    print("=" * 80)
    try:
        driver_perf = analyzer.analyze_driver_performance_by_track_type()
        
        if not driver_perf.empty:
            print(f"✓ Success: Analyzed {len(driver_perf)} driver-track_type combinations")
            print(f"  Drivers: {driver_perf['driver_number'].nunique()}")
            print(f"  Track types: {driver_perf['track_type'].unique().tolist()}")
            print(f"\n  Sample data:")
            print(driver_perf.head())
        else:
            print("✗ No data returned")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test 2: Track Difficulty Rankings
    print("=" * 80)
    print("Test 2: Track Difficulty Rankings")
    print("=" * 80)
    try:
        difficulty_df = analyzer.create_track_difficulty_rankings()
        
        if not difficulty_df.empty:
            print(f"✓ Success: Ranked {len(difficulty_df)} tracks")
            print(f"\n  Top 3 most difficult:")
            for idx, row in difficulty_df.head(3).iterrows():
                print(f"    {row['difficulty_rank']}. {row['track_id']} (score: {row['difficulty_score']:.3f})")
        else:
            print("✗ No rankings generated")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test 3: Track-Specific Patterns
    print("=" * 80)
    print("Test 3: Track-Specific Patterns")
    print("=" * 80)
    try:
        patterns = analyzer.identify_track_specific_patterns("barber", "Race 1")
        
        if patterns:
            print(f"✓ Success: Identified patterns for {patterns.get('track_id', 'unknown')}")
            print(f"  Patterns found: {list(patterns.keys())}")
        else:
            print("✗ No patterns identified")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test 4: Complete Report
    print("=" * 80)
    print("Test 4: Complete Performance Report")
    print("=" * 80)
    try:
        report = analyzer.generate_performance_report()
        
        if report:
            print(f"✓ Success: Generated report")
            print(f"  Components:")
            print(f"    - Driver performance: {len(report.get('driver_performance_by_type', []))} records")
            print(f"    - Difficulty rankings: {len(report.get('track_difficulty_rankings', []))} tracks")
            print(f"    - Track patterns: {len(report.get('track_patterns', {}))} tracks")
            
            if 'summary' in report:
                print(f"\n  Summary:")
                summary = report['summary']
                print(f"    Total drivers: {summary.get('total_drivers_analyzed', 0)}")
        else:
            print("✗ No report generated")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)

def test_convenience_function():
    """Test convenience function"""
    print("\n" + "=" * 80)
    print("TESTING CONVENIENCE FUNCTION")
    print("=" * 80)
    print()
    
    loader = DataLoader()
    
    print("Testing analyze_track_performance()...")
    try:
        report = analyze_track_performance(data_loader=loader)
        if report:
            print(f"✓ analyze_track_performance() successful")
            print(f"  Report components: {list(report.keys())}")
        else:
            print("✗ analyze_track_performance() returned empty result")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    test_performance_analyzer()
    test_convenience_function()

