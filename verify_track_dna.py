"""
Quick verification of Track DNA Extractor
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("VERIFYING TRACK DNA EXTRACTOR")
print("=" * 80)

try:
    from src.data_processing.track_dna_extractor import TrackDNAExtractor, extract_all_tracks_dna
    from src.data_processing.data_loader import DataLoader
    
    print("✓ Imports successful")
    
    loader = DataLoader()
    extractor = TrackDNAExtractor(loader)
    
    print(f"✓ Extractor initialized")
    print(f"  Venues: {', '.join(loader.venues)}")
    
    # Test extraction
    print("\nTesting DNA extraction for barber Race 1...")
    dna = extractor.extract_track_dna("barber", "Race 1")
    
    print("✓ DNA extraction successful")
    print(f"\nExtracted Features:")
    print(f"  - Complexity Score: {dna['technical_complexity']['complexity_score']:.3f}")
    print(f"  - Number of Sectors: {dna['physical_characteristics']['num_sectors']}")
    print(f"  - Track Length: {dna['physical_characteristics']['track_length'].get('estimated_length_km', 0):.3f} km")
    
    if dna['speed_profile']['top_speed']:
        print(f"  - Top Speed: {dna['speed_profile']['top_speed'].get('max', 0):.1f} km/h")
    
    print("\n✓ All tests passed!")
    print("=" * 80)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

