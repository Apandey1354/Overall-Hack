"""
Test script for Track DNA Extractor
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.data_loader import DataLoader
from src.data_processing.track_dna_extractor import TrackDNAExtractor, extract_all_tracks_dna

def test_track_dna():
    """Test the Track DNA extractor"""
    print("=" * 80)
    print("TESTING TRACK DNA EXTRACTOR")
    print("=" * 80)
    print()
    
    # Initialize
    loader = DataLoader()
    extractor = TrackDNAExtractor(loader)
    
    print(f"Venues available: {', '.join(loader.venues)}")
    print()
    
    # Test extraction for each venue
    for venue in loader.venues[:2]:  # Test first 2 venues
        print(f"\n{'=' * 80}")
        print(f"Testing {venue.upper()}")
        print("=" * 80)
        
        try:
            dna = extractor.extract_track_dna(venue, "Race 1")
            
            print(f"\n✓ Track DNA extracted successfully")
            print(f"\nTechnical Complexity Score: {dna['technical_complexity']['complexity_score']:.3f}")
            print(f"  - Sector variance: {dna['technical_complexity'].get('overall_sector_variance', 0):.3f}")
            print(f"  - Braking zones: {dna['technical_complexity']['braking_zones'].get('count', 0)}")
            
            print(f"\nSpeed Profile:")
            if dna['speed_profile']['top_speed']:
                print(f"  - Top speed: {dna['speed_profile']['top_speed'].get('max', 0):.1f} km/h")
            print(f"  - Straight/Corner ratio: {dna['speed_profile']['straight_corner_ratio'].get('ratio', 0):.2f}")
            
            print(f"\nPhysical Characteristics:")
            track_len = dna['physical_characteristics']['track_length']
            print(f"  - Track length: {track_len.get('estimated_length_km', 0):.3f} km")
            print(f"  - Number of sectors: {dna['physical_characteristics']['num_sectors']}")
            
            print(f"\nPerformance Patterns:")
            if dna['performance_patterns']['lap_time_variance']:
                lap_var = dna['performance_patterns']['lap_time_variance']
                print(f"  - Lap time CV: {lap_var.get('cv', 0):.3f}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test extracting all tracks
    print(f"\n{'=' * 80}")
    print("EXTRACTING DNA FOR ALL TRACKS")
    print("=" * 80)
    try:
        all_dna_df = extract_all_tracks_dna(loader)
        print(f"✓ Extracted DNA for {len(all_dna_df)} track/race combinations")
        print(f"\nColumns: {list(all_dna_df.columns)}")
        print(f"\nSample row:")
        if len(all_dna_df) > 0:
            print(all_dna_df.iloc[0].to_dict())
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_track_dna()

