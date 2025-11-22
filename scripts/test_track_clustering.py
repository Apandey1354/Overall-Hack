"""
Test script for Track Clustering & Classification
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.track_clustering import TrackClusterer, cluster_all_tracks, build_track_similarity_matrix
from src.data_processing.data_loader import DataLoader

def test_clustering():
    """Test the Track Clustering system"""
    print("=" * 80)
    print("TESTING TRACK CLUSTERING & CLASSIFICATION")
    print("=" * 80)
    print()
    
    # Initialize
    loader = DataLoader()
    clusterer = TrackClusterer(data_loader=loader)
    
    print(f"Venues available: {', '.join(loader.venues)}")
    print()
    
    # Step 1: Extract features
    print("Step 1: Extracting Track DNA features...")
    features_df = clusterer.extract_features_for_clustering()
    
    if features_df.empty:
        print("✗ No features extracted")
        return
    
    print(f"✓ Extracted features for {len(features_df)} tracks")
    print(f"  Features: {len([c for c in features_df.columns if c not in ['venue', 'race', 'track_id']])} numeric features")
    print()
    
    # Step 2: Cluster tracks
    print("Step 2: Clustering tracks...")
    clustered_df = clusterer.cluster_tracks(features_df=features_df, n_clusters=None)
    
    if 'cluster' in clustered_df.columns:
        print(f"✓ Clustering complete")
        print(f"  Clusters found: {clustered_df['cluster'].nunique()}")
        print(f"\n  Cluster distribution:")
        cluster_counts = clustered_df['cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            tracks = clustered_df[clustered_df['cluster'] == cluster]['track_id'].tolist()
            print(f"    Cluster {cluster}: {count} tracks - {', '.join(tracks[:3])}{'...' if len(tracks) > 3 else ''}")
    print()
    
    # Step 3: Classify tracks
    print("Step 3: Classifying tracks...")
    classified_df = clusterer.classify_tracks(clustered_df)
    
    if 'classification' in classified_df.columns:
        print(f"✓ Classification complete")
        print(f"\n  Classification distribution:")
        class_counts = classified_df['classification'].value_counts()
        for cls, count in class_counts.items():
            print(f"    {cls:20s}: {count} tracks")
        
        print(f"\n  Track classifications:")
        for idx, row in classified_df.iterrows():
            print(f"    {row['track_id']:30s}: {row['classification']:15s} "
                  f"(tech={row.get('technical_score', 0):.2f}, speed={row.get('speed_score', 0):.2f})")
    print()
    
    # Step 4: Build similarity matrix
    print("Step 4: Building track similarity matrix...")
    similarity_matrix = clusterer.build_similarity_matrix(features_df=features_df)
    
    if not similarity_matrix.empty:
        print(f"✓ Similarity matrix built: {similarity_matrix.shape}")
        print(f"\n  Sample similarities (first 3 tracks):")
        track_ids = similarity_matrix.index[:3]
        for track_id in track_ids:
            similar = similarity_matrix.loc[track_id].nlargest(3)
            print(f"    {track_id}:")
            for other_track, sim in similar.items():
                if other_track != track_id:
                    print(f"      - {other_track}: {sim:.3f}")
    print()
    
    # Step 5: Cluster summary
    print("Step 5: Cluster summary...")
    summary = clusterer.get_cluster_summary(classified_df)
    if not summary.empty:
        print(f"✓ Cluster summary generated")
        print(f"\n  Summary statistics:")
        print(summary.head())
    print()
    
    print("=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)

def test_convenience_functions():
    """Test convenience functions"""
    print("\n" + "=" * 80)
    print("TESTING CONVENIENCE FUNCTIONS")
    print("=" * 80)
    print()
    
    loader = DataLoader()
    
    # Test cluster_all_tracks
    print("Testing cluster_all_tracks()...")
    try:
        result = cluster_all_tracks(data_loader=loader)
        if not result.empty:
            print(f"✓ cluster_all_tracks() successful: {len(result)} tracks")
            print(f"  Columns: {list(result.columns)}")
        else:
            print("✗ cluster_all_tracks() returned empty result")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test build_track_similarity_matrix
    print("Testing build_track_similarity_matrix()...")
    try:
        sim_matrix = build_track_similarity_matrix(data_loader=loader)
        if not sim_matrix.empty:
            print(f"✓ build_track_similarity_matrix() successful: {sim_matrix.shape}")
        else:
            print("✗ build_track_similarity_matrix() returned empty result")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    test_clustering()
    test_convenience_functions()

