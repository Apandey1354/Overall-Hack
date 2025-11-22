"""
Test script for Driver Embedder
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.driver_embedder import DriverEmbedder, create_driver_embeddings
from src.data_processing.data_loader import DataLoader
import numpy as np

def test_driver_embedder():
    """Test the Driver Embedder"""
    print("=" * 80)
    print("TESTING DRIVER EMBEDDER")
    print("=" * 80)
    print()
    
    # Initialize
    loader = DataLoader()
    embedder = DriverEmbedder(data_loader=loader)
    
    print(f"Venues available: {', '.join(loader.venues)}")
    print()
    
    # Test 1: Create embedding for a single driver
    print("=" * 80)
    print("Test 1: Create Embedding for Single Driver")
    print("=" * 80)
    
    # Get a driver number from results
    driver_num = None
    for venue in loader.venues[:1]:
        results_df = loader.load_results_file(venue, "Race 1", "provisional")
        if results_df.empty:
            results_df = loader.load_results_file(venue, "Race 1", "official")
        if not results_df.empty and 'NUMBER' in results_df.columns:
            driver_num = int(results_df.iloc[0]['NUMBER'])
            break
    
    if driver_num:
        try:
            embedding = embedder.create_driver_embedding(driver_num)
            
            print(f"✓ Success: Created embedding for driver #{driver_num}")
            print(f"\n  Skill Vector Shape: {embedding['overall_skill_vector'].shape}")
            print(f"  Skill Vector: {embedding['overall_skill_vector']}")
            
            print(f"\n  Technical Proficiency: {embedding['technical_proficiency']['score']:.3f}")
            print(f"  High-Speed Proficiency: {embedding['high_speed_proficiency']['score']:.3f}")
            print(f"  Consistency Score: {embedding['consistency_metrics']['consistency_score']:.3f}")
            print(f"  Weather Adaptability: {embedding['weather_adaptability']['adaptability_score']:.3f}")
            
            strengths = embedding['track_specific_strengths']
            print(f"\n  Track-Specific Strengths:")
            print(f"    Best Track Type: {strengths.get('best_track_type', 'N/A')}")
            print(f"    Strengths: {', '.join(strengths.get('strengths', []))}")
            print(f"    Weaknesses: {', '.join(strengths.get('weaknesses', []))}")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ No driver found to test")
    
    print()
    
    # Test 2: Create embeddings for all drivers
    print("=" * 80)
    print("Test 2: Create Embeddings for All Drivers")
    print("=" * 80)
    try:
        embeddings_df = embedder.create_all_driver_embeddings()
        
        if not embeddings_df.empty:
            print(f"✓ Success: Created embeddings for {len(embeddings_df)} drivers")
            print(f"\n  Columns: {list(embeddings_df.columns)}")
            print(f"\n  Sample embeddings:")
            print(embeddings_df[['driver_number', 'technical_proficiency', 'high_speed_proficiency', 
                                'consistency_score', 'best_track_type']].head())
            
            print(f"\n  Summary Statistics:")
            print(f"    Avg Technical Proficiency: {embeddings_df['technical_proficiency'].mean():.3f}")
            print(f"    Avg High-Speed Proficiency: {embeddings_df['high_speed_proficiency'].mean():.3f}")
            print(f"    Avg Consistency: {embeddings_df['consistency_score'].mean():.3f}")
        else:
            print("✗ No embeddings created")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test 3: Skill vector analysis
    print("=" * 80)
    print("Test 3: Skill Vector Analysis")
    print("=" * 80)
    try:
        embeddings_df = embedder.create_all_driver_embeddings()
        
        if not embeddings_df.empty and 'skill_vector' in embeddings_df.columns:
            # Extract skill vectors
            skill_vectors = np.array([v for v in embeddings_df['skill_vector'] if v is not None])
            
            if len(skill_vectors) > 0:
                print(f"✓ Success: Analyzed {len(skill_vectors)} skill vectors")
                print(f"  Vector Dimension: {skill_vectors.shape[1]}")
                print(f"  Mean Vector: {skill_vectors.mean(axis=0)}")
                print(f"  Std Vector: {skill_vectors.std(axis=0)}")
        else:
            print("✗ No skill vectors available")
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
    
    print("Testing create_driver_embeddings()...")
    try:
        embeddings_df = create_driver_embeddings(data_loader=loader)
        if not embeddings_df.empty:
            print(f"✓ create_driver_embeddings() successful: {len(embeddings_df)} drivers")
            print(f"  Columns: {list(embeddings_df.columns)}")
        else:
            print("✗ create_driver_embeddings() returned empty result")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    import numpy as np
    test_driver_embedder()
    test_convenience_function()

