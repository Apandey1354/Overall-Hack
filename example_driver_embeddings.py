"""
Example: Driver Embedding Creation
Run this script to see driver skill vector creation in action
"""

import sys
from pathlib import Path
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing.driver_embedder import DriverEmbedder, create_driver_embeddings
from src.data_processing.data_loader import DataLoader

print("=" * 80)
print("DRIVER EMBEDDING CREATION EXAMPLE")
print("=" * 80)
print()

# Initialize
print("Step 1: Initializing...")
loader = DataLoader()
embedder = DriverEmbedder(data_loader=loader)
print(f"✓ Initialized")
print(f"  Available venues: {', '.join(loader.venues)}")
print()

# Example 1: Create embedding for a single driver
print("=" * 80)
print("Example 1: Create Embedding for Single Driver")
print("=" * 80)

# Get a driver number
driver_num = None
for venue in loader.venues:
    results_df = loader.load_results_file(venue, "Race 1", "provisional")
    if results_df.empty:
        results_df = loader.load_results_file(venue, "Race 1", "official")
    if not results_df.empty and 'NUMBER' in results_df.columns:
        driver_num = int(results_df.iloc[0]['NUMBER'])
        break

if driver_num:
    try:
        embedding = embedder.create_driver_embedding(driver_num)
        
        print(f"\n✓ Created embedding for driver #{driver_num}")
        print(f"\n📊 Skill Components:")
        print(f"  Technical Proficiency: {embedding['technical_proficiency']['score']:.3f}")
        if embedding['technical_proficiency'].get('avg_position'):
            print(f"    - Avg Position at Technical Tracks: {embedding['technical_proficiency']['avg_position']:.1f}")
        
        print(f"  High-Speed Proficiency: {embedding['high_speed_proficiency']['score']:.3f}")
        if embedding['high_speed_proficiency'].get('avg_position'):
            print(f"    - Avg Position at Speed Tracks: {embedding['high_speed_proficiency']['avg_position']:.1f}")
        
        print(f"  Consistency Score: {embedding['consistency_metrics']['consistency_score']:.3f}")
        print(f"    - Lap Time CV: {embedding['consistency_metrics']['lap_time_cv_avg']:.4f}")
        print(f"    - Finish Rate: {embedding['consistency_metrics']['finish_rate']:.2%}")
        
        print(f"  Weather Adaptability: {embedding['weather_adaptability']['adaptability_score']:.3f}")
        if embedding['weather_adaptability'].get('rain_performance'):
            print(f"    - Rain Performance: {embedding['weather_adaptability']['rain_performance']:.3f}")
        if embedding['weather_adaptability'].get('dry_performance'):
            print(f"    - Dry Performance: {embedding['weather_adaptability']['dry_performance']:.3f}")
        
        strengths = embedding['track_specific_strengths']
        print(f"\n  Track-Specific Profile:")
        print(f"    Best Track Type: {strengths.get('best_track_type', 'N/A')}")
        print(f"    Strengths: {', '.join(strengths.get('strengths', ['None']))}")
        print(f"    Weaknesses: {', '.join(strengths.get('weaknesses', ['None']))}")
        
        print(f"\n🎯 Overall Skill Vector (8-dimensional):")
        skill_vector = embedding['overall_skill_vector']
        print(f"  Shape: {skill_vector.shape}")
        print(f"  Values: {skill_vector}")
        print(f"  Components:")
        print(f"    [0] Technical Proficiency: {skill_vector[0]:.3f}")
        print(f"    [1] High-Speed Proficiency: {skill_vector[1]:.3f}")
        print(f"    [2] Consistency: {skill_vector[2]:.3f}")
        print(f"    [3] Weather Adaptability: {skill_vector[3]:.3f}")
        print(f"    [4] Technical Track Score: {skill_vector[4]:.3f}")
        print(f"    [5] Speed Track Score: {skill_vector[5]:.3f}")
        print(f"    [6] Balanced Track Score: {skill_vector[6]:.3f}")
        print(f"    [7] Finish Rate: {skill_vector[7]:.3f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("✗ No driver found")

# Example 2: Create embeddings for all drivers
print("\n" + "=" * 80)
print("Example 2: Create Embeddings for All Drivers")
print("=" * 80)

try:
    embeddings_df = create_driver_embeddings(data_loader=loader)
    
    if not embeddings_df.empty:
        print(f"\n✓ Created embeddings for {len(embeddings_df)} drivers")
        print(f"\n📊 Summary Statistics:")
        print(f"  Technical Proficiency:")
        print(f"    Mean: {embeddings_df['technical_proficiency'].mean():.3f}")
        print(f"    Std: {embeddings_df['technical_proficiency'].std():.3f}")
        print(f"    Max: {embeddings_df['technical_proficiency'].max():.3f}")
        
        print(f"\n  High-Speed Proficiency:")
        print(f"    Mean: {embeddings_df['high_speed_proficiency'].mean():.3f}")
        print(f"    Std: {embeddings_df['high_speed_proficiency'].std():.3f}")
        print(f"    Max: {embeddings_df['high_speed_proficiency'].max():.3f}")
        
        print(f"\n  Consistency:")
        print(f"    Mean: {embeddings_df['consistency_score'].mean():.3f}")
        print(f"    Std: {embeddings_df['consistency_score'].std():.3f}")
        
        print(f"\n🏆 Top Performers:")
        print(f"\n  Top Technical Drivers:")
        top_technical = embeddings_df.nlargest(3, 'technical_proficiency')
        for idx, row in top_technical.iterrows():
            print(f"    Driver #{int(row['driver_number']):3d}: {row['technical_proficiency']:.3f}")
        
        print(f"\n  Top Speed Drivers:")
        top_speed = embeddings_df.nlargest(3, 'high_speed_proficiency')
        for idx, row in top_speed.iterrows():
            print(f"    Driver #{int(row['driver_number']):3d}: {row['high_speed_proficiency']:.3f}")
        
        print(f"\n  Most Consistent Drivers:")
        top_consistent = embeddings_df.nlargest(3, 'consistency_score')
        for idx, row in top_consistent.iterrows():
            print(f"    Driver #{int(row['driver_number']):3d}: {row['consistency_score']:.3f}")
        
        print(f"\n📈 Track Type Specializations:")
        track_type_counts = embeddings_df['best_track_type'].value_counts()
        for track_type, count in track_type_counts.items():
            print(f"  {track_type:20s}: {count} drivers")
    else:
        print("✗ No embeddings created")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Example 3: Skill vector similarity
print("\n" + "=" * 80)
print("Example 3: Skill Vector Similarity Analysis")
print("=" * 80)

try:
    embeddings_df = create_driver_embeddings(data_loader=loader)
    
    if not embeddings_df.empty and 'skill_vector' in embeddings_df.columns:
        # Extract skill vectors
        skill_vectors = []
        driver_nums = []
        for idx, row in embeddings_df.iterrows():
            if row['skill_vector'] is not None:
                skill_vectors.append(row['skill_vector'])
                driver_nums.append(row['driver_number'])
        
        if len(skill_vectors) >= 2:
            skill_vectors = np.array(skill_vectors)
            
            print(f"\n✓ Analyzing {len(skill_vectors)} skill vectors")
            
            # Calculate pairwise cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            
            similarity_matrix = cosine_similarity(skill_vectors)
            
            # Find most similar drivers
            print(f"\n  Most Similar Driver Pairs:")
            pairs = []
            for i in range(len(driver_nums)):
                for j in range(i+1, len(driver_nums)):
                    sim = similarity_matrix[i, j]
                    pairs.append((driver_nums[i], driver_nums[j], sim))
            
            pairs.sort(key=lambda x: x[2], reverse=True)
            
            for driver1, driver2, sim in pairs[:5]:
                print(f"    Driver #{driver1:3d} <-> Driver #{driver2:3d}: {sim:.3f}")
    else:
        print("✗ Not enough skill vectors for similarity analysis")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("EXAMPLES COMPLETE")
print("=" * 80)

# Save results to file
print("\nSaving results to driver_embeddings_results.txt...")
try:
    with open("driver_embeddings_results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DRIVER EMBEDDINGS RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Get all embeddings
        embeddings_df = create_driver_embeddings(data_loader=loader)
        
        if not embeddings_df.empty:
            f.write(f"Total Drivers: {len(embeddings_df)}\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Technical Proficiency:\n")
            f.write(f"  Mean: {embeddings_df['technical_proficiency'].mean():.3f}\n")
            f.write(f"  Std:  {embeddings_df['technical_proficiency'].std():.3f}\n")
            f.write(f"  Min:  {embeddings_df['technical_proficiency'].min():.3f}\n")
            f.write(f"  Max:  {embeddings_df['technical_proficiency'].max():.3f}\n\n")
            
            f.write(f"High-Speed Proficiency:\n")
            f.write(f"  Mean: {embeddings_df['high_speed_proficiency'].mean():.3f}\n")
            f.write(f"  Std:  {embeddings_df['high_speed_proficiency'].std():.3f}\n")
            f.write(f"  Min:  {embeddings_df['high_speed_proficiency'].min():.3f}\n")
            f.write(f"  Max:  {embeddings_df['high_speed_proficiency'].max():.3f}\n\n")
            
            f.write(f"Consistency Score:\n")
            f.write(f"  Mean: {embeddings_df['consistency_score'].mean():.3f}\n")
            f.write(f"  Std:  {embeddings_df['consistency_score'].std():.3f}\n")
            f.write(f"  Min:  {embeddings_df['consistency_score'].min():.3f}\n")
            f.write(f"  Max:  {embeddings_df['consistency_score'].max():.3f}\n\n")
            
            f.write(f"Weather Adaptability:\n")
            f.write(f"  Mean: {embeddings_df['weather_adaptability'].mean():.3f}\n")
            f.write(f"  Std:  {embeddings_df['weather_adaptability'].std():.3f}\n")
            f.write(f"  Min:  {embeddings_df['weather_adaptability'].min():.3f}\n")
            f.write(f"  Max:  {embeddings_df['weather_adaptability'].max():.3f}\n\n")
            
            # Top performers
            f.write("\n" + "=" * 80 + "\n")
            f.write("TOP PERFORMERS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Top 10 Technical Drivers:\n")
            f.write("-" * 80 + "\n")
            top_technical = embeddings_df.nlargest(10, 'technical_proficiency')
            for idx, row in top_technical.iterrows():
                f.write(f"  Driver #{int(row['driver_number']):3d}: "
                       f"Technical={row['technical_proficiency']:.3f}, "
                       f"Speed={row['high_speed_proficiency']:.3f}, "
                       f"Consistency={row['consistency_score']:.3f}, "
                       f"Best Track Type={row['best_track_type']}\n")
            
            f.write("\nTop 10 Speed Drivers:\n")
            f.write("-" * 80 + "\n")
            top_speed = embeddings_df.nlargest(10, 'high_speed_proficiency')
            for idx, row in top_speed.iterrows():
                f.write(f"  Driver #{int(row['driver_number']):3d}: "
                       f"Technical={row['technical_proficiency']:.3f}, "
                       f"Speed={row['high_speed_proficiency']:.3f}, "
                       f"Consistency={row['consistency_score']:.3f}, "
                       f"Best Track Type={row['best_track_type']}\n")
            
            f.write("\nTop 10 Most Consistent Drivers:\n")
            f.write("-" * 80 + "\n")
            top_consistent = embeddings_df.nlargest(10, 'consistency_score')
            for idx, row in top_consistent.iterrows():
                f.write(f"  Driver #{int(row['driver_number']):3d}: "
                       f"Technical={row['technical_proficiency']:.3f}, "
                       f"Speed={row['high_speed_proficiency']:.3f}, "
                       f"Consistency={row['consistency_score']:.3f}, "
                       f"Best Track Type={row['best_track_type']}\n")
            
            # Track type specializations
            f.write("\n" + "=" * 80 + "\n")
            f.write("TRACK TYPE SPECIALIZATIONS\n")
            f.write("=" * 80 + "\n\n")
            track_type_counts = embeddings_df['best_track_type'].value_counts()
            for track_type, count in track_type_counts.items():
                f.write(f"  {track_type:20s}: {count} drivers\n")
            
            # All driver embeddings with skill vectors
            f.write("\n" + "=" * 80 + "\n")
            f.write("ALL DRIVER EMBEDDINGS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"{'Driver':<8} {'Tech':<8} {'Speed':<8} {'Consist':<8} {'Weather':<8} {'Best Type':<15} {'Strengths':<30}\n")
            f.write("-" * 80 + "\n")
            
            for idx, row in embeddings_df.iterrows():
                strengths = row.get('strengths', '')[:28]  # Truncate if too long
                f.write(f"{int(row['driver_number']):<8} "
                       f"{row['technical_proficiency']:<8.3f} "
                       f"{row['high_speed_proficiency']:<8.3f} "
                       f"{row['consistency_score']:<8.3f} "
                       f"{row['weather_adaptability']:<8.3f} "
                       f"{str(row['best_track_type']):<15} "
                       f"{strengths:<30}\n")
            
            # Skill vectors
            f.write("\n" + "=" * 80 + "\n")
            f.write("SKILL VECTORS (8-dimensional)\n")
            f.write("=" * 80 + "\n")
            f.write("Format: [Technical, High-Speed, Consistency, Weather, Tech-Track, Speed-Track, Balanced-Track, Finish-Rate]\n")
            f.write("-" * 80 + "\n\n")
            
            for idx, row in embeddings_df.iterrows():
                if row['skill_vector'] is not None:
                    vector = row['skill_vector']
                    f.write(f"Driver #{int(row['driver_number']):3d}:\n")
                    f.write(f"  [{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}, {vector[3]:.3f}, "
                           f"{vector[4]:.3f}, {vector[5]:.3f}, {vector[6]:.3f}, {vector[7]:.3f}]\n")
            
            # Similarity analysis
            f.write("\n" + "=" * 80 + "\n")
            f.write("DRIVER SIMILARITY (Top 10 Most Similar Pairs)\n")
            f.write("=" * 80 + "\n\n")
            
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                
                skill_vectors = []
                driver_nums = []
                for idx, row in embeddings_df.iterrows():
                    if row['skill_vector'] is not None:
                        skill_vectors.append(row['skill_vector'])
                        driver_nums.append(row['driver_number'])
                
                if len(skill_vectors) >= 2:
                    skill_vectors = np.array(skill_vectors)
                    similarity_matrix = cosine_similarity(skill_vectors)
                    
                    pairs = []
                    for i in range(len(driver_nums)):
                        for j in range(i+1, len(driver_nums)):
                            sim = similarity_matrix[i, j]
                            pairs.append((driver_nums[i], driver_nums[j], sim))
                    
                    pairs.sort(key=lambda x: x[2], reverse=True)
                    
                    for driver1, driver2, sim in pairs[:10]:
                        f.write(f"  Driver #{driver1:3d} <-> Driver #{driver2:3d}: {sim:.3f}\n")
            except Exception as e:
                f.write(f"  Similarity analysis failed: {e}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF RESULTS\n")
        f.write("=" * 80 + "\n")
    
    print("✓ Results saved to driver_embeddings_results.txt")
except Exception as e:
    print(f"✗ Error saving results: {e}")
    import traceback
    traceback.print_exc()

print("\nNext steps:")
print("1. Use skill vectors for transfer learning models")
print("2. Predict driver performance on new tracks")
print("3. Build driver similarity matrix for recommendations")

