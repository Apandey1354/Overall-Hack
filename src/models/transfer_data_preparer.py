"""
Data Preparation for Transfer Learning Model
Prepares training data from driver embeddings, track DNA, and performance data
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TransferDataPreparer:
    """
    Prepares data for transfer learning model training.
    Creates source-target track pairs with driver performance data.
    """
    
    def __init__(self, data_loader=None):
        """
        Initialize data preparer.
        
        Args:
            data_loader: DataLoader instance (optional)
        """
        if data_loader is None:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from src.data_processing.data_loader import DataLoader
            self.loader = DataLoader()
        else:
            self.loader = data_loader
    
    def prepare_training_pairs(
        self,
        driver_embeddings_df: pd.DataFrame,
        track_dna_df: pd.DataFrame,
        min_races_per_driver: int = 2
    ) -> pd.DataFrame:
        """
        Prepare source-target track pairs for training.
        Creates pairs where we have driver performance at both tracks.
        
        Args:
            driver_embeddings_df: DataFrame with driver embeddings
            track_dna_df: DataFrame with track DNA features
            min_races_per_driver: Minimum races a driver must have to be included
            
        Returns:
            DataFrame with training pairs
        """
        logger.info("Preparing training pairs...")
        
        training_pairs = []
        
        # Get all unique track IDs
        track_ids = track_dna_df['track_id'].unique() if 'track_id' in track_dna_df.columns else []
        
        # For each driver, create pairs of tracks they've raced at
        for driver_num in driver_embeddings_df['driver_number'].unique():
            driver_embedding = driver_embeddings_df[
                driver_embeddings_df['driver_number'] == driver_num
            ].iloc[0]
            
            # Get tracks this driver has raced at
            driver_tracks = []
            for venue in self.loader.venues:
                for race in ["Race 1", "Race 2"]:
                    track_id = f"{venue}_{race}"
                    results_df = self.loader.load_results_file(venue, race, "provisional")
                    if results_df.empty:
                        results_df = self.loader.load_results_file(venue, race, "official")
                    
                    if not results_df.empty:
                        driver_result = results_df[results_df['NUMBER'] == driver_num]
                        if not driver_result.empty:
                            # Get performance metrics
                            position = driver_result.iloc[0].get('POSITION', None)
                            fastest_lap_time = driver_result.iloc[0].get('FL_TIME', None)
                            fastest_lap_speed = driver_result.iloc[0].get('FL_KPH', None)
                            laps = driver_result.iloc[0].get('LAPS', None)
                            finished = driver_result.iloc[0].get('STATUS', '').lower() not in ['dnf', 'dsq', 'nc']
                            
                            # Parse fastest lap time
                            fastest_lap_seconds = None
                            if fastest_lap_time and isinstance(fastest_lap_time, str):
                                try:
                                    parts = fastest_lap_time.split(':')
                                    if len(parts) == 2:
                                        minutes = int(parts[0])
                                        seconds_parts = parts[1].split('.')
                                        seconds = int(seconds_parts[0])
                                        milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
                                        fastest_lap_seconds = minutes * 60 + seconds + milliseconds / 1000
                                except:
                                    pass
                            
                            driver_tracks.append({
                                'track_id': track_id,
                                'venue': venue,
                                'race': race,
                                'position': float(position) if pd.notna(position) else None,
                                'fastest_lap_time': fastest_lap_seconds,
                                'fastest_lap_speed': float(fastest_lap_speed) if pd.notna(fastest_lap_speed) else None,
                                'laps': int(laps) if pd.notna(laps) else None,
                                'finished': finished
                            })
            
            # Create pairs: each track as source, others as target
            if len(driver_tracks) >= min_races_per_driver:
                for i, source_track in enumerate(driver_tracks):
                    for j, target_track in enumerate(driver_tracks):
                        if i != j:  # Don't pair track with itself
                            # Get track DNA for both tracks
                            source_dna = track_dna_df[track_dna_df['track_id'] == source_track['track_id']]
                            target_dna = track_dna_df[track_dna_df['track_id'] == target_track['track_id']]
                            
                            if not source_dna.empty and not target_dna.empty:
                                # Prepare performance features (normalized)
                                source_perf = self._extract_performance_features(source_track, include_laps_feature=True)
                                target_perf = self._extract_performance_features(target_track, include_laps_feature=False)
                                
                                training_pairs.append({
                                    'driver_number': driver_num,
                                    'source_track_id': source_track['track_id'],
                                    'target_track_id': target_track['track_id'],
                                    'source_performance': source_perf,
                                    'target_performance': target_perf,  # Ground truth
                                    'source_dna': source_dna.iloc[0].to_dict(),
                                    'target_dna': target_dna.iloc[0].to_dict()
                                })
        
        pairs_df = pd.DataFrame(training_pairs)
        logger.info(f"Created {len(pairs_df)} training pairs")
        
        return pairs_df
    
    def _extract_performance_features(self, track_data: Dict, include_laps_feature: bool = True) -> np.ndarray:
        """
        Extract normalized performance features.
        
        Args:
            track_data: Dictionary with performance data
            
        Returns:
            Normalized performance features array
        """
        features = []
        
        # Position (normalized: 1st = 1.0, 20th = 0.05)
        position = track_data.get('position', 20)
        if position:
            pos_score = max(0, 1.0 - (position - 1) / 19)
        else:
            pos_score = 0.0
        features.append(pos_score)
        
        # Fastest lap time (normalized by typical range: 90-160 seconds)
        lap_time = track_data.get('fastest_lap_time', None)
        if lap_time:
            # Normalize: faster = higher score
            lap_score = max(0, min(1, (160 - lap_time) / 70))
        else:
            lap_score = 0.0
        features.append(lap_score)
        
        # Fastest lap speed (normalized: 150-200 km/h range)
        speed = track_data.get('fastest_lap_speed', None)
        if speed:
            speed_score = max(0, min(1, (speed - 150) / 50))
        else:
            speed_score = 0.0
        features.append(speed_score)
        
        # Finish status (0 or 1)
        finished = 1.0 if track_data.get('finished', False) else 0.0
        features.append(finished)

        if include_laps_feature:
            # Laps completed (normalized: 0-30 range)
            laps = track_data.get('laps', 0)
            if laps:
                laps_score = min(1.0, laps / 30)
            else:
                laps_score = 0.0
            features.append(laps_score)
        
        return np.array(features, dtype=np.float32)
    
    def prepare_tensors(
        self,
        training_pairs_df: pd.DataFrame,
        driver_embeddings_df: pd.DataFrame,
        track_dna_df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert training pairs to PyTorch tensors.
        
        Args:
            training_pairs_df: DataFrame with training pairs
            driver_embeddings_df: DataFrame with driver embeddings
            track_dna_df: DataFrame with track DNA features
            
        Returns:
            Tuple of (driver_embeddings, source_performance, target_dna, target_performance)
        """
        logger.info("Converting to tensors...")
        
        driver_embeddings_list = []
        source_perf_list = []
        target_dna_list = []
        target_perf_list = []
        
        for idx, row in training_pairs_df.iterrows():
            # Get driver embedding
            driver_num = row['driver_number']
            driver_embedding = driver_embeddings_df[
                driver_embeddings_df['driver_number'] == driver_num
            ]
            
            if not driver_embedding.empty and 'skill_vector' in driver_embedding.columns:
                skill_vector = driver_embedding.iloc[0]['skill_vector']
                if skill_vector is not None:
                    driver_embeddings_list.append(skill_vector)
                else:
                    continue
            else:
                continue
            
            # Get source performance
            source_perf = row.get('source_performance', np.zeros(5, dtype=np.float32))
            if isinstance(source_perf, np.ndarray):
                source_perf_list.append(source_perf.astype(np.float32))
            else:
                source_perf_list.append(np.zeros(5, dtype=np.float32))
            
            # Get target track DNA (flatten from dict)
            target_dna_dict = row.get('target_dna', {})
            target_dna_vector = self._flatten_track_dna(target_dna_dict, track_dna_df)
            target_dna_list.append(target_dna_vector)
            
            # Get target performance (ground truth)
            target_perf = row.get('target_performance', np.zeros(4, dtype=np.float32))
            if isinstance(target_perf, np.ndarray):
                target_perf_list.append(target_perf.astype(np.float32))
            else:
                target_perf_list.append(np.zeros(4, dtype=np.float32))
        
        # Convert to tensors
        driver_tensor = torch.FloatTensor(np.array(driver_embeddings_list))
        source_tensor = torch.FloatTensor(np.array(source_perf_list))
        target_dna_tensor = torch.FloatTensor(np.array(target_dna_list))
        target_perf_tensor = torch.FloatTensor(np.array(target_perf_list))
        
        logger.info(f"Prepared tensors: {driver_tensor.shape}, {source_tensor.shape}, "
                   f"{target_dna_tensor.shape}, {target_perf_tensor.shape}")
        
        return driver_tensor, source_tensor, target_dna_tensor, target_perf_tensor
    
    def _flatten_track_dna(self, dna_dict: Dict, track_dna_df: pd.DataFrame) -> np.ndarray:
        """
        Flatten track DNA dictionary to feature vector.
        
        Args:
            dna_dict: Track DNA dictionary
            track_dna_df: Reference DataFrame for structure
            
        Returns:
            Flattened DNA feature vector
        """
        features = []
        
        # Extract key features from DNA
        tech = dna_dict.get('technical_complexity', {})
        if isinstance(tech, dict):
            features.append(tech.get('complexity_score', 0.0))
            sector_std = tech.get('overall_sector_std', tech.get('overall_sector_variance', 0.0))
            features.append(sector_std)
            features.append(tech.get('braking_zones', {}).get('count', 0) / 20.0)  # Normalize
        
        speed = dna_dict.get('speed_profile', {})
        if isinstance(speed, dict):
            top_speed = speed.get('top_speed', {})
            if isinstance(top_speed, dict):
                features.append(top_speed.get('max', 0.0) / 200.0)  # Normalize
            else:
                features.append(0.0)
            features.append(speed.get('straight_corner_ratio', {}).get('ratio', 0.0) / 2.0)  # Normalize
        
        physical = dna_dict.get('physical_characteristics', {})
        if isinstance(physical, dict):
            track_len = physical.get('track_length', {})
            if isinstance(track_len, dict):
                features.append(track_len.get('estimated_length_km', 0.0) / 10.0)  # Normalize
            else:
                features.append(0.0)
            features.append(physical.get('num_sectors', 0) / 10.0)  # Normalize
        
        patterns = dna_dict.get('performance_patterns', {})
        if isinstance(patterns, dict):
            lap_var = patterns.get('lap_time_variance', {})
            if isinstance(lap_var, dict):
                features.append(lap_var.get('cv', 0.0) * 10)  # Normalize
            else:
                features.append(0.0)
        
        # Pad to fixed size (20 features)
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20], dtype=np.float32)

