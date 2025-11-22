"""
Example: Transfer Learning Model for Track Performance Prediction
Run this script to see transfer learning in action
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.transfer_learning_model import TrackTransferModel, create_transfer_model, TransferLearningTrainer
from src.models.transfer_data_preparer import TransferDataPreparer
from src.data_processing.driver_embedder import create_driver_embeddings
from src.data_processing.track_dna_extractor import extract_all_tracks_dna
from src.data_processing.data_loader import DataLoader

report_lines = []


def log(line: str = "", store: bool = True):
    print(line)
    if store:
        report_lines.append(line)


log("=" * 80)
log("TRANSFER LEARNING MODEL EXAMPLE")
log("=" * 80)
log()

# Initialize
log("Step 1: Initializing...")
loader = DataLoader()
log(f"✓ Initialized")
log(f"  Available venues: {', '.join(loader.venues)}")
log()

# Step 1: Get driver embeddings
log("=" * 80)
log("Step 1: Creating Driver Embeddings")
log("=" * 80)

try:
    driver_embeddings_df = create_driver_embeddings(data_loader=loader)
    
    if not driver_embeddings_df.empty:
        log(f"✓ Created embeddings for {len(driver_embeddings_df)} drivers")
        log(f"  Skill vector dimension: {driver_embeddings_df.iloc[0]['skill_vector'].shape}")
    else:
        log("✗ No driver embeddings created")
        sys.exit(1)
except Exception as e:
    log(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Get track DNA
log("\n" + "=" * 80)
log("Step 2: Extracting Track DNA")
log("=" * 80)

try:
    track_dna_df = extract_all_tracks_dna(loader)
    
    if not track_dna_df.empty:
        log(f"✓ Extracted DNA for {len(track_dna_df)} tracks")
    else:
        log("✗ No track DNA extracted")
        sys.exit(1)
except Exception as e:
    log(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Prepare training data
log("\n" + "=" * 80)
log("Step 3: Preparing Training Data")
log("=" * 80)

try:
    preparer = TransferDataPreparer(data_loader=loader)
    training_pairs_df = preparer.prepare_training_pairs(
        driver_embeddings_df=driver_embeddings_df,
        track_dna_df=track_dna_df,
        min_races_per_driver=2
    )
    
    if not training_pairs_df.empty:
        log(f"✓ Created {len(training_pairs_df)} training pairs")
        log(f"  Unique drivers: {training_pairs_df['driver_number'].nunique()}")
        log(f"  Unique track pairs: {training_pairs_df[['source_track_id', 'target_track_id']].drop_duplicates().shape[0]}")
    else:
        log("✗ No training pairs created")
        sys.exit(1)
except Exception as e:
    log(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Convert to tensors
log("\n" + "=" * 80)
log("Step 4: Converting to Tensors")
log("=" * 80)

try:
    driver_tensor, source_tensor, target_dna_tensor, target_perf_tensor = preparer.prepare_tensors(
        training_pairs_df=training_pairs_df,
        driver_embeddings_df=driver_embeddings_df,
        track_dna_df=track_dna_df
    )
    
    log(f"✓ Converted to tensors")
    log(f"  Driver embeddings: {driver_tensor.shape}")
    log(f"  Source performance: {source_tensor.shape}")
    log(f"  Target DNA: {target_dna_tensor.shape}")
    log(f"  Target performance: {target_perf_tensor.shape}")
except Exception as e:
    log(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Create model
log("\n" + "=" * 80)
nlog = log  # alias if needed later
log("Step 5: Creating Transfer Learning Model")
log("=" * 80)

try:
    model = create_transfer_model(
        driver_embedding_dim=8,
        track_dna_dim=20,
        performance_features=5
    )
    
    log(f"✓ Model created")
    log(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    log(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    if len(driver_tensor) > 0:
        model.eval()
        with torch.no_grad():
            sample_driver = driver_tensor[0:1]
            sample_source = source_tensor[0:1]
            sample_target = target_dna_tensor[0:1]
            
            prediction = model(sample_driver, sample_source, sample_target)
            log(f"\n  Sample prediction shape: {prediction.shape}")
            log(f"  Sample prediction: {prediction[0].numpy()}")
            log(f"    [Lap Time, Position, Speed, Finish Probability]")
except Exception as e:
    log(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Example prediction
log("\n" + "=" * 80)
log("Step 6: Example Prediction")
log("=" * 80)

try:
    # Get a sample driver and tracks
    sample_driver = driver_embeddings_df.iloc[0]
    driver_num = sample_driver['driver_number']
    
    # Find tracks this driver raced at
    source_track_id = None
    target_track_id = None
    
    for venue in loader.venues:
        for race in ["Race 1", "Race 2"]:
            track_id = f"{venue}_{race}"
            results_df = loader.load_results_file(venue, race, "provisional")
            if results_df.empty:
                results_df = loader.load_results_file(venue, race, "official")
            
            if not results_df.empty:
                driver_result = results_df[results_df['NUMBER'] == driver_num]
                if not driver_result.empty:
                    if source_track_id is None:
                        source_track_id = track_id
                    elif target_track_id is None:
                        target_track_id = track_id
                        break
        if target_track_id:
            break
    
    if source_track_id and target_track_id:
        # Get source performance
        source_venue, source_race = source_track_id.split('_', 1)
        results_df = loader.load_results_file(source_venue, source_race, "provisional")
        if results_df.empty:
            results_df = loader.load_results_file(source_venue, source_race, "official")
        
        driver_result = results_df[results_df['NUMBER'] == driver_num]
        if not driver_result.empty:
            # Prepare inputs
            driver_embedding = sample_driver['skill_vector']
            
            # Get source performance features
            position = driver_result.iloc[0].get('POSITION', 20)
            fl_time = driver_result.iloc[0].get('FL_TIME', None)
            fl_speed = driver_result.iloc[0].get('FL_KPH', None)
            laps = driver_result.iloc[0].get('LAPS', 0)
            finished = driver_result.iloc[0].get('STATUS', '').lower() not in ['dnf', 'dsq', 'nc']
            
            # Parse lap time
            lap_time_seconds = None
            if fl_time and isinstance(fl_time, str):
                try:
                    parts = fl_time.split(':')
                    if len(parts) == 2:
                        minutes = int(parts[0])
                        seconds_parts = parts[1].split('.')
                        seconds = int(seconds_parts[0])
                        milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
                        lap_time_seconds = minutes * 60 + seconds + milliseconds / 1000
                except:
                    pass
            
            # Normalize features
            pos_score = max(0, 1.0 - (position - 1) / 19) if position else 0
            lap_score = max(0, min(1, (160 - lap_time_seconds) / 70)) if lap_time_seconds else 0
            speed_score = max(0, min(1, (fl_speed - 150) / 50)) if fl_speed else 0
            laps_score = min(1.0, laps / 30) if laps else 0
            finish_score = 1.0 if finished else 0.0
            
            source_perf = np.array([pos_score, lap_score, speed_score, laps_score, finish_score])
            
            # Get target track DNA
            target_dna_row = track_dna_df[track_dna_df['track_id'] == target_track_id]
            if not target_dna_row.empty:
                target_dna_dict = target_dna_row.iloc[0].to_dict()
                target_dna = preparer._flatten_track_dna(target_dna_dict, track_dna_df)
                
                # Make prediction
                prediction = model.predict(
                    driver_embedding=driver_embedding,
                    source_performance=source_perf,
                    target_track_dna=target_dna
                )
                
                log(f"\n✓ Prediction for Driver #{driver_num}")
                log(f"  Source Track: {source_track_id}")
                log(f"  Target Track: {target_track_id}")
                log(f"\n  Predicted Performance:")
                log(f"    Lap Time Score: {prediction[0]:.3f}")
                log(f"    Position Score: {prediction[1]:.3f}")
                log(f"    Speed Score: {prediction[2]:.3f}")
                log(f"    Finish Probability: {prediction[3]:.3f}")
                
                # Get actual performance for comparison
                target_venue, target_race = target_track_id.split('_', 1)
                target_results = loader.load_results_file(target_venue, target_race, "provisional")
                if target_results.empty:
                    target_results = loader.load_results_file(target_venue, target_race, "official")
                
                if not target_results.empty:
                    actual_result = target_results[target_results['NUMBER'] == driver_num]
                    if not actual_result.empty:
                        actual_pos = actual_result.iloc[0].get('POSITION', None)
                        log(f"\n  Actual Performance:")
                        log(f"    Position: {actual_pos}")
except Exception as e:
    log(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

log("\n" + "=" * 80)
log("EXAMPLE COMPLETE")
log("=" * 80)
log("\nNext steps:")
log("1. Train the model on all track pairs")
log("2. Validate on held-out tracks")
log("3. Use for performance prediction on new tracks")

log("\nSaving results to transfer_learning_results.txt...")
try:
    results_path = Path("transfer_learning_results.txt")
    with results_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"✓ Results saved to {results_path.resolve()}")
except Exception as e:
    log(f"✗ Error saving results: {e}")
    import traceback
    traceback.print_exc()

