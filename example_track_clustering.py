"""
Example: How to use Track Clustering & Classification
Run this script to see track clustering in action
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing.track_clustering import TrackClusterer, cluster_all_tracks, build_track_similarity_matrix
from src.data_processing.data_loader import DataLoader

print("=" * 80)
print("TRACK CLUSTERING & CLASSIFICATION EXAMPLE")
print("=" * 80)
print()

# Initialize
print("Step 1: Initializing...")
loader = DataLoader()
clusterer = TrackClusterer(data_loader=loader)
print(f"✓ Initialized")
print(f"  Available venues: {', '.join(loader.venues)}")
print()

# Example 1: Cluster all tracks
print("=" * 80)
print("Example 1: Cluster All Tracks")
print("=" * 80)

try:
    # Use convenience function
    clustered_df = cluster_all_tracks(data_loader=loader)
    
    if not clustered_df.empty:
        print(f"\n✓ Clustered {len(clustered_df)} tracks")
        print(f"\nCluster Distribution:")
        cluster_counts = clustered_df['cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            print(f"  Cluster {cluster}: {count} tracks")
        
        print(f"\nTrack Classifications:")
        if 'classification' in clustered_df.columns:
            for idx, row in clustered_df.iterrows():
                print(f"  {row['track_id']:30s}: {row['classification']:15s} "
                      f"(Cluster {row['cluster']})")
        
        print(f"\nClassification Summary:")
        if 'classification' in clustered_df.columns:
            class_counts = clustered_df['classification'].value_counts()
            for cls, count in class_counts.items():
                print(f"  {cls:20s}: {count} tracks")
    else:
        print("✗ No tracks clustered")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Example 2: Build similarity matrix
print("\n" + "=" * 80)
print("Example 2: Track Similarity Matrix")
print("=" * 80)

try:
    similarity_matrix = build_track_similarity_matrix(data_loader=loader)
    
    if not similarity_matrix.empty:
        print(f"\n✓ Similarity matrix built: {similarity_matrix.shape}")
        print(f"\nMost Similar Track Pairs:")
        
        # Find most similar pairs (excluding self-similarity)
        track_ids = similarity_matrix.index.tolist()
        pairs = []
        for i, track1 in enumerate(track_ids):
            for track2 in track_ids[i+1:]:
                sim = similarity_matrix.loc[track1, track2]
                pairs.append((track1, track2, sim))
        
        # Sort by similarity
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Show top 5
        for track1, track2, sim in pairs[:5]:
            print(f"  {track1:30s} <-> {track2:30s}: {sim:.3f}")
    else:
        print("✗ No similarity matrix generated")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Example 3: Custom clustering
print("\n" + "=" * 80)
print("Example 3: Custom Clustering (3 clusters)")
print("=" * 80)

try:
    clusterer = TrackClusterer(data_loader=loader)
    features_df = clusterer.extract_features_for_clustering()
    
    if not features_df.empty:
        clustered_df = clusterer.cluster_tracks(n_clusters=3, features_df=features_df)
        classified_df = clusterer.classify_tracks(clustered_df)
        
        print(f"\n✓ Custom clustering complete")
        print(f"\nCluster Summary:")
        summary = clusterer.get_cluster_summary(classified_df)
        if not summary.empty:
            print(summary)
    else:
        print("✗ No features extracted")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("EXAMPLES COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("1. Use clusters for track-specific performance analysis")
print("2. Build transfer learning models based on track similarity")
print("3. Create track recommendation system")

