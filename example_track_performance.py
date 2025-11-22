"""
Example: Track-Specific Performance Analysis
Run this script to see track performance analysis in action
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing.track_performance_analyzer import TrackPerformanceAnalyzer, analyze_track_performance
from src.data_processing.data_loader import DataLoader

print("=" * 80)
print("TRACK-SPECIFIC PERFORMANCE ANALYSIS EXAMPLE")
print("=" * 80)
print()

# Initialize
print("Step 1: Initializing...")
loader = DataLoader()
analyzer = TrackPerformanceAnalyzer(data_loader=loader)
print(f"✓ Initialized")
print(f"  Available venues: {', '.join(loader.venues)}")
print()

# Example 1: Driver Performance by Track Type
print("=" * 80)
print("Example 1: Driver Performance by Track Type")
print("=" * 80)

try:
    driver_perf = analyzer.analyze_driver_performance_by_track_type()
    
    if not driver_perf.empty:
        print(f"\n✓ Analyzed performance for {driver_perf['driver_number'].nunique()} drivers")
        print(f"  Total driver-track_type combinations: {len(driver_perf)}")
        
        print(f"\nTop Performers by Track Type:")
        for track_type in ['technical', 'speed-focused', 'balanced']:
            type_data = driver_perf[driver_perf['track_type'] == track_type]
            if not type_data.empty:
                top = type_data.nlargest(3, 'performance_score')
                print(f"\n  {track_type.upper()}:")
                for idx, row in top.iterrows():
                    print(f"    Driver #{int(row['driver_number']):3d}: "
                          f"Avg Position: {row['avg_position']:.1f}, "
                          f"Performance Score: {row['performance_score']:.3f}")
        
        print(f"\nDriver Specializations:")
        # Find drivers who excel at specific track types
        for driver_num in driver_perf['driver_number'].unique()[:5]:  # Show first 5 drivers
            driver_data = driver_perf[driver_perf['driver_number'] == driver_num]
            if len(driver_data) > 0:
                best_type = driver_data.loc[driver_data['performance_score'].idxmax()]
                print(f"  Driver #{int(driver_num):3d}: Best at {best_type['track_type']:15s} "
                      f"(Score: {best_type['performance_score']:.3f}, "
                      f"Avg Pos: {best_type['avg_position']:.1f})")
    else:
        print("✗ No driver performance data found")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Example 2: Track Difficulty Rankings
print("\n" + "=" * 80)
print("Example 2: Track Difficulty Rankings")
print("=" * 80)

try:
    difficulty_df = analyzer.create_track_difficulty_rankings()
    
    if not difficulty_df.empty:
        print(f"\n✓ Created difficulty rankings for {len(difficulty_df)} tracks")
        print(f"\nMost Difficult Tracks (Top 5):")
        top_difficult = difficulty_df.head(5)
        for idx, row in top_difficult.iterrows():
            print(f"  {row['difficulty_rank']:2d}. {row['track_id']:30s} "
                  f"(Score: {row['difficulty_score']:.3f}, "
                  f"Type: {row['track_classification']:15s})")
        
        print(f"\nEasiest Tracks (Bottom 5):")
        bottom_easy = difficulty_df.tail(5)
        for idx, row in bottom_easy.iterrows():
            print(f"  {row['difficulty_rank']:2d}. {row['track_id']:30s} "
                  f"(Score: {row['difficulty_score']:.3f}, "
                  f"Type: {row['track_classification']:15s})")
        
        print(f"\nDifficulty by Track Type:")
        difficulty_by_type = difficulty_df.groupby('track_classification')['difficulty_score'].agg(['mean', 'std'])
        for track_type, row in difficulty_by_type.iterrows():
            print(f"  {track_type:20s}: Mean = {row['mean']:.3f}, Std = {row['std']:.3f}")
    else:
        print("✗ No difficulty rankings generated")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Example 3: Track-Specific Patterns
print("\n" + "=" * 80)
print("Example 3: Track-Specific Driving Patterns")
print("=" * 80)

try:
    # Analyze patterns for a specific track
    patterns = analyzer.identify_track_specific_patterns("barber", "Race 1")
    
    if patterns:
        print(f"\n✓ Identified patterns for {patterns['track_id']}")
        
        if 'sector_patterns' in patterns:
            sector = patterns['sector_patterns']
            print(f"\n  Sector Patterns:")
            print(f"    Most Variable Sector: {sector.get('most_variable_sector', 'N/A')}")
            print(f"    Least Variable Sector: {sector.get('least_variable_sector', 'N/A')}")
            print(f"    Variance Ratio: {sector.get('sector_variance_ratio', 0):.2f}")
        
        if 'speed_patterns' in patterns:
            speed = patterns['speed_patterns']
            print(f"\n  Speed Patterns:")
            print(f"    Speed Range: {speed.get('speed_range', 0):.1f} km/h")
            print(f"    Speed Consistency: {speed.get('speed_consistency', 0):.3f}")
            print(f"    High Speed Laps Ratio: {speed.get('high_speed_laps_ratio', 0):.2%}")
        
        if 'lap_progression' in patterns:
            prog = patterns['lap_progression']
            print(f"\n  Lap Progression:")
            print(f"    Improvement Over Race: {prog.get('improvement_over_race', 0):.3f} seconds")
            print(f"    Shows Improvement: {prog.get('shows_improvement', False)}")
        
        if 'competitiveness' in patterns:
            comp = patterns['competitiveness']
            print(f"\n  Competitiveness:")
            print(f"    Position Spread: {comp.get('position_spread', 0):.1f}")
            print(f"    Close Competition: {comp.get('close_competition', False)}")
    else:
        print("✗ No patterns identified")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Example 4: Complete Performance Report
print("\n" + "=" * 80)
print("Example 4: Complete Performance Report")
print("=" * 80)

try:
    report = analyzer.generate_performance_report()
    
    if report:
        print(f"\n✓ Generated comprehensive performance report")
        
        if 'summary' in report:
            summary = report['summary']
            print(f"\n  Summary:")
            print(f"    Total Drivers Analyzed: {summary.get('total_drivers_analyzed', 0)}")
            print(f"    Drivers by Track Type: {summary.get('drivers_by_track_type', {})}")
        
        if 'track_difficulty_rankings' in report and not report['track_difficulty_rankings'].empty:
            print(f"\n  Track Difficulty Rankings: {len(report['track_difficulty_rankings'])} tracks")
        
        if 'track_patterns' in report:
            print(f"\n  Track Patterns Analyzed: {len(report['track_patterns'])} tracks")
    else:
        print("✗ No report generated")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("EXAMPLES COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Use driver performance data for transfer learning models")
print("2. Build driver skill embeddings based on track type performance")
print("3. Create track recommendation system for driver training")

