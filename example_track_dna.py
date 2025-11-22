"""
Example: How to use Track DNA Extractor
Run this script to see Track DNA extraction in action
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing.track_dna_extractor import TrackDNAExtractor, extract_all_tracks_dna
from src.data_processing.data_loader import DataLoader

print("=" * 80)
print("TRACK DNA EXTRACTION EXAMPLE")
print("=" * 80)
print()

# Initialize
print("Step 1: Initializing DataLoader and TrackDNAExtractor...")
loader = DataLoader()
extractor = TrackDNAExtractor(loader)
print(f"✓ Initialized")
print(f"  Available venues: {', '.join(loader.venues)}")
print()

# Example 1: Extract DNA for a single track
print("=" * 80)
print("Example 1: Extract DNA for Barber Race 1")
print("=" * 80)
try:
    dna = extractor.extract_track_dna("barber", "Race 1")
    
    print("\n📊 Technical Complexity:")
    complexity = dna['technical_complexity']
    print(f"  Complexity Score: {complexity['complexity_score']:.3f} (0-1 scale)")
    print(f"  Sector Variance: {complexity.get('overall_sector_variance', 0):.3f}")
    print(f"  Braking Zones: {complexity['braking_zones'].get('count', 0)}")
    
    print("\n🏎️ Speed Profile:")
    speed = dna['speed_profile']
    if speed['top_speed']:
        print(f"  Top Speed: {speed['top_speed'].get('max', 0):.1f} km/h")
        print(f"  Average Speed: {speed['speed_distribution'].get('mean', 0):.1f} km/h")
    print(f"  Straight/Corner Ratio: {speed['straight_corner_ratio'].get('ratio', 0):.2f}")
    
    print("\n📏 Physical Characteristics:")
    physical = dna['physical_characteristics']
    track_len = physical['track_length']
    print(f"  Track Length: {track_len.get('estimated_length_km', 0):.3f} km")
    print(f"  Number of Sectors: {physical['num_sectors']}")
    print(f"  Sector Names: {', '.join(physical['sector_names'])}")
    
    print("\n📈 Performance Patterns:")
    patterns = dna['performance_patterns']
    if patterns['lap_time_variance']:
        lap_var = patterns['lap_time_variance']
        print(f"  Lap Time CV: {lap_var.get('cv', 0):.3f}")
        print(f"  Lap Time Range: {lap_var.get('range', 0):.2f} seconds")
    
    print("\n✓ DNA extraction successful!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Example 2: Extract DNA for all tracks
print("\n" + "=" * 80)
print("Example 2: Extract DNA for ALL Tracks")
print("=" * 80)
try:
    all_dna_df = extract_all_tracks_dna(loader)
    
    print(f"\n✓ Extracted DNA for {len(all_dna_df)} track/race combinations")
    print(f"\nDataFrame shape: {all_dna_df.shape}")
    print(f"Columns: {list(all_dna_df.columns)}")
    
    # Show summary
    if len(all_dna_df) > 0:
        print("\nSummary by Venue:")
        for venue in loader.venues:
            venue_data = all_dna_df[all_dna_df['venue'] == venue]
            if len(venue_data) > 0:
                # Get complexity scores
                complexities = []
                for idx, row in venue_data.iterrows():
                    comp = row['technical_complexity']
                    if isinstance(comp, dict) and 'complexity_score' in comp:
                        complexities.append(comp['complexity_score'])
                
                if complexities:
                    avg_comp = sum(complexities) / len(complexities)
                    print(f"  {venue:30s}: Avg Complexity = {avg_comp:.3f}")
    
    print("\n✓ All tracks DNA extracted successfully!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("EXAMPLES COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Use DNA features for track clustering (Phase 1.2)")
print("2. Build track similarity matrix")
print("3. Analyze driver performance by track type")

