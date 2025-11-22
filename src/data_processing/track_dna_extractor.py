"""
Track DNA Extractor
Extracts unique characteristics and features from racing track data.

Based on Phase 1.1 of the project plan:
- Technical Complexity Score
- Speed Profile
- Physical Characteristics
- Performance Patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

KNOWN_TRACK_LENGTHS_KM = {
    "barber": 3.83,
    "cota": 5.51,
    "circuit of the americas": 5.51,
    "indianapolis": 3.925,
    "virginia-international-raceway": 5.26,
    "vir": 5.26,
}


class TrackDNAExtractor:
    """
    Extract track DNA features from racing data.
    
    Track DNA Components:
    1. Technical Complexity Score
    2. Speed Profile
    3. Physical Characteristics
    4. Performance Patterns
    """
    
    def __init__(self, data_loader=None):
        """
        Initialize the Track DNA Extractor.
        
        Args:
            data_loader: DataLoader instance (optional, will create if not provided)
        """
        if data_loader is None:
            from .data_loader import DataLoader
            self.loader = DataLoader()
        else:
            self.loader = data_loader

        # Cache full telemetry loads per venue/race so multiple feature builders
        # reuse the same (complete) dataset instead of re-reading truncated samples.
        self._telemetry_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    
    def extract_track_dna(self, venue: str, race: str = "Race 1") -> Dict:
        """
        Extract complete track DNA for a venue and race.
        
        Args:
            venue: Track venue name
            race: Race number ("Race 1" or "Race 2")
            
        Returns:
            Dictionary containing all track DNA features
        """
        logger.info(f"Extracting track DNA for {venue} - {race}")
        
        track_id = f"{venue}_{race}"

        dna = {
            'track_id': track_id,
            'venue': venue,
            'race': race,
            'technical_complexity': self.extract_technical_complexity(venue, race),
            'speed_profile': self.extract_speed_profile(venue, race),
            'physical_characteristics': self.extract_physical_characteristics(venue, race),
            'performance_patterns': self.extract_performance_patterns(venue, race)
        }
        
        return dna
    
    def extract_technical_complexity(self, venue: str, race: str) -> Dict:
        """
        Extract Technical Complexity Score features.
        
        Features:
        - Sector time variance
        - Number of braking zones
        - Average corner speeds
        
        Args:
            venue: Track venue name
            race: Race number
            
        Returns:
            Dictionary with technical complexity metrics
        """
        logger.info(f"Extracting technical complexity for {venue} - {race}")
        
        # Load analysis data for sector times
        analysis_df = self.loader.load_analysis_endurance(venue, race)
        
        if analysis_df.empty:
            logger.warning(f"No analysis data for {venue} - {race}")
            return self._empty_complexity_dict()
        
        complexity = {}
        
        # 1. Sector time variance
        sector_cols = ['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']
        available_sectors = [col for col in sector_cols if col in analysis_df.columns]
        
        if available_sectors:
            sector_variances = {}
            sector_std_values = []
            for col in available_sectors:
                sector_data = analysis_df[col].dropna()
                if len(sector_data) > 0:
                    sector_name = col.replace('_SECONDS', '')
                    mean_time = sector_data.mean()
                    std_time = sector_data.std()
                    variance = std_time ** 2
                    sector_std_values.append(std_time)
                    normalized = float(np.clip(std_time / max(0.1, mean_time * 0.6), 0, 1))
                    sector_variances[sector_name] = {
                        'variance': float(variance),
                        'std_seconds': float(std_time),
                        'mean_seconds': float(mean_time),
                        'cv': float(std_time / mean_time) if mean_time > 0 else 0,
                        'normalized_variance': normalized
                    }
            
            complexity['sector_time_variance'] = sector_variances
            if sector_std_values:
                complexity['overall_sector_std'] = float(np.mean(sector_std_values))
                complexity['overall_sector_variance'] = float(np.mean([std ** 2 for std in sector_std_values]))
            else:
                complexity['overall_sector_std'] = 0.0
                complexity['overall_sector_variance'] = 0.0
        else:
            complexity['sector_time_variance'] = {}
            complexity['overall_sector_variance'] = 0.0
            complexity['overall_sector_std'] = 0.0
        
        # 2. Number of braking zones (from telemetry)
        braking_zones = self._count_braking_zones(venue, race)
        complexity['braking_zones'] = braking_zones
        
        # 3. Average corner speeds (from sector analysis and telemetry)
        corner_speeds = self._extract_corner_speeds(venue, race, analysis_df)
        complexity['corner_speeds'] = corner_speeds
        
        # 4. Additional telemetry-based complexity features
        telemetry_features = self._extract_comprehensive_telemetry_features(venue, race)
        if telemetry_features.get('available', False):
            complexity['telemetry_features'] = {
                'acceleration_variance': telemetry_features.get('acceleration', {}).get('std', 0),
                'steering_activity': telemetry_features.get('steering', {}).get('std', 0),
                'gear_changes': telemetry_features.get('gear', {}).get('std', 0),  # Higher std = more gear changes
                'throttle_variance': telemetry_features.get('throttle', {}).get('std', 0)
            }
        else:
            complexity['telemetry_features'] = {}
        
        # Overall complexity score (normalized 0-1)
        complexity['complexity_score'] = self._calculate_complexity_score(complexity)
        
        return complexity
    
    def extract_speed_profile(self, venue: str, race: str) -> Dict:
        """
        Extract Speed Profile features.
        
        Features:
        - Top speed zones
        - Average speed distribution
        - Straight vs. corner ratio
        
        Args:
            venue: Track venue name
            race: Race number
            
        Returns:
            Dictionary with speed profile metrics
        """
        logger.info(f"Extracting speed profile for {venue} - {race}")
        
        analysis_df = self.loader.load_analysis_endurance(venue, race)
        
        if analysis_df.empty:
            logger.warning(f"No analysis data for {venue} - {race}")
            return self._empty_speed_profile_dict()
        
        profile = {}
        
        # 1. Top speed zones
        if 'TOP_SPEED' in analysis_df.columns:
            top_speeds = analysis_df['TOP_SPEED'].dropna()
            if len(top_speeds) > 0:
                profile['top_speed'] = {
                    'max': float(top_speeds.max()),
                    'mean': float(top_speeds.mean()),
                    'median': float(top_speeds.median()),
                    'std': float(top_speeds.std()),
                    'p95': float(top_speeds.quantile(0.95)),
                    'p5': float(top_speeds.quantile(0.05))
                }
            else:
                profile['top_speed'] = {}
        else:
            profile['top_speed'] = {}
        
        # 2. Average speed distribution
        if 'KPH' in analysis_df.columns:
            speeds = analysis_df['KPH'].dropna()
            if len(speeds) > 0:
                profile['speed_distribution'] = {
                    'mean': float(speeds.mean()),
                    'median': float(speeds.median()),
                    'std': float(speeds.std()),
                    'min': float(speeds.min()),
                    'max': float(speeds.max()),
                    'q25': float(speeds.quantile(0.25)),
                    'q75': float(speeds.quantile(0.75))
                }
            else:
                profile['speed_distribution'] = {}
        else:
            profile['speed_distribution'] = {}
        
        # 3. Straight vs. corner ratio (from sector times)
        straight_corner_ratio = self._calculate_straight_corner_ratio(analysis_df)
        profile['straight_corner_ratio'] = straight_corner_ratio
        
        # 4. Speed from telemetry (if available, more granular than analysis data)
        telemetry_features = self._extract_comprehensive_telemetry_features(venue, race)
        if telemetry_features.get('available', False) and 'speed' in telemetry_features:
            speed_data = telemetry_features['speed']
            profile['telemetry_speed'] = {
                'mean': speed_data.get('mean', 0),
                'max': speed_data.get('max', 0),
                'std': speed_data.get('std', 0)
            }
        else:
            profile['telemetry_speed'] = {}
        
        return profile
    
    def extract_physical_characteristics(self, venue: str, race: str) -> Dict:
        """
        Extract Physical Characteristics features.
        
        Features:
        - Track length
        - Elevation changes
        - Number of sectors
        
        Args:
            venue: Track venue name
            race: Race number
            
        Returns:
            Dictionary with physical characteristics
        """
        logger.info(f"Extracting physical characteristics for {venue} - {race}")
        
        analysis_df = self.loader.load_analysis_endurance(venue, race)
        
        if analysis_df.empty:
            logger.warning(f"No analysis data for {venue} - {race}")
            return self._empty_physical_dict()
        
        characteristics = {}
        
        # 1. Track length (estimated from lap times and speeds)
        track_length = self._estimate_track_length(venue, race, analysis_df)
        characteristics['track_length'] = track_length
        
        # 2. Elevation changes (from telemetry if available)
        elevation = self._extract_elevation_data(venue, race)
        characteristics['elevation'] = elevation
        
        # 3. Number of sectors
        sector_cols = [col for col in analysis_df.columns if col.startswith('S') and col.endswith('_SECONDS')]
        characteristics['num_sectors'] = len(sector_cols)
        characteristics['sector_names'] = [col.replace('_SECONDS', '') for col in sector_cols]
        
        # Additional: Intermediate timing points
        intermediate_cols = [col for col in analysis_df.columns if col.startswith('IM')]
        characteristics['num_intermediate_points'] = len(set([col.split('_')[0] for col in intermediate_cols]))
        
        return characteristics
    
    def extract_performance_patterns(self, venue: str, race: str) -> Dict:
        """
        Extract Performance Patterns features.
        
        Features:
        - Lap time variance
        - Sector consistency
        - Overtaking opportunities
        
        Args:
            venue: Track venue name
            race: Race number
            
        Returns:
            Dictionary with performance pattern metrics
        """
        logger.info(f"Extracting performance patterns for {venue} - {race}")
        
        analysis_df = self.loader.load_analysis_endurance(venue, race)
        results_df = self.loader.load_results_file(venue, race, "provisional")
        
        if analysis_df.empty:
            logger.warning(f"No analysis data for {venue} - {race}")
            return self._empty_performance_dict()
        
        patterns = {}
        
        # 1. Lap time variance
        if 'LAP_TIME_seconds' in analysis_df.columns:
            lap_times = analysis_df['LAP_TIME_seconds'].dropna()
            if len(lap_times) > 0:
                patterns['lap_time_variance'] = {
                    'variance': float(lap_times.var()),
                    'std': float(lap_times.std()),
                    'mean': float(lap_times.mean()),
                    'cv': float(lap_times.std() / lap_times.mean()) if lap_times.mean() > 0 else 0,
                    'min': float(lap_times.min()),
                    'max': float(lap_times.max()),
                    'range': float(lap_times.max() - lap_times.min())
                }
            else:
                patterns['lap_time_variance'] = {}
        else:
            patterns['lap_time_variance'] = {}
        
        # 2. Sector consistency (coefficient of variation per sector)
        sector_consistency = self._calculate_sector_consistency(analysis_df)
        patterns['sector_consistency'] = sector_consistency
        
        # 3. Overtaking opportunities (from position changes in results)
        overtaking = self._analyze_overtaking_opportunities(venue, race, results_df, analysis_df)
        patterns['overtaking_opportunities'] = overtaking
        
        return patterns
    
    # Helper methods
    
    def _extract_comprehensive_telemetry_features(self, venue: str, race: str) -> Dict:
        """
        Extract comprehensive features from telemetry data.
        Uses all available telemetry parameters for richer analysis.
        """
        try:
            telemetry = self._get_full_telemetry(venue, race)
            
            if telemetry.empty or 'parameter_name' not in telemetry.columns:
                return {'available': False, 'method': 'telemetry_unavailable'}
            
            features = {'available': True, 'parameters_found': []}
            
            # Parse timestamps for time-based analysis
            if 'timestamp_parsed' in telemetry.columns:
                telemetry = telemetry.sort_values('timestamp_parsed')
            
            # Extract each parameter type
            param_groups = {
                'brake': ['pbrake_f', 'pbrake_r', 'brake'],
                'acceleration': ['accx_can', 'accy_can', 'acc'],
                'throttle': ['ath', 'throttle'],
                'steering': ['Steering_Angle', 'steering', 'steer'],
                'gear': ['gear'],
                'speed': ['speed', 'kph', 'velocity'],
                'rpm': ['rpm', 'engine']
            }
            
            for group_name, patterns in param_groups.items():
                # Find matching parameters
                matching = telemetry[telemetry['parameter_name'].str.contains(
                    '|'.join(patterns), case=False, na=False, regex=True
                )]
                
                if not matching.empty:
                    values = pd.to_numeric(matching['value'], errors='coerce').dropna()
                    if len(values) > 0:
                        features[group_name] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'median': float(values.median()),
                            'count': int(len(values))
                        }
                        features['parameters_found'].extend(matching['parameter_name'].unique().tolist())
            
            # Remove duplicates from parameters_found
            features['parameters_found'] = list(set(features['parameters_found']))
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting comprehensive telemetry features: {e}")
            return {'available': False, 'method': 'error', 'error': str(e)}
    
    def _count_braking_zones(self, venue: str, race: str) -> Dict:
        """Count braking zones from telemetry data using comprehensive analysis."""
        try:
            telemetry_features = self._extract_comprehensive_telemetry_features(venue, race)
            
            if not telemetry_features.get('available', False):
                return {'count': 0, 'method': 'telemetry_unavailable'}
            
            # Use brake data if available
            if 'brake' in telemetry_features:
                brake_data = telemetry_features['brake']
                # More sophisticated braking zone detection
                # Count significant brake pressure peaks per lap
                telemetry = self._get_full_telemetry(venue, race)
                
                if not telemetry.empty and 'lap' in telemetry.columns:
                    brake_params = telemetry[telemetry['parameter_name'].str.contains('brake', case=False, na=False)]
                    if not brake_params.empty:
                        brake_values = pd.to_numeric(brake_params['value'], errors='coerce').dropna()
                        if len(brake_values) > 0:
                            threshold = brake_values.quantile(0.8)
                            
                            # Group by lap and count significant braking events
                            brake_params['value_numeric'] = pd.to_numeric(brake_params['value'], errors='coerce')
                            brake_params['is_braking'] = brake_params['value_numeric'] > threshold
                            
                            if 'lap' in brake_params.columns:
                                braking_by_lap = brake_params.groupby('lap')['is_braking'].sum()
                                avg_braking_events_per_lap = braking_by_lap.mean()
                                
                                # Estimate zones: significant braking events that are distinct
                                # Use change detection to find distinct braking zones
                                estimated_zones = max(1, int(avg_braking_events_per_lap / 50))
                                
                                return {
                                    'count': estimated_zones,
                                    'method': 'comprehensive_telemetry_analysis',
                                    'avg_brake_pressure': float(brake_data['mean']),
                                    'max_brake_pressure': float(brake_data['max']),
                                    'braking_events_per_lap': float(avg_braking_events_per_lap)
                                }
                
                return {
                    'count': max(1, int(brake_data['max'] / 100)),  # Heuristic fallback
                    'method': 'telemetry_analysis',
                    'avg_brake_pressure': float(brake_data['mean']),
                    'max_brake_pressure': float(brake_data['max'])
                }
            
            return {'count': 0, 'method': 'no_brake_data'}
            
        except Exception as e:
            logger.warning(f"Error counting braking zones: {e}")
            return {'count': 0, 'method': 'error', 'error': str(e)}
    
    def _extract_corner_speeds(self, venue: str, race: str, analysis_df: pd.DataFrame) -> Dict:
        """Extract average corner speeds."""
        corner_speeds = {}
        
        # Use sector times and speeds to estimate corner speeds
        # Slower sectors likely have more corners
        if 'KPH' in analysis_df.columns and 'S1_SECONDS' in analysis_df.columns:
            speeds = analysis_df['KPH'].dropna()
            sector_times = analysis_df[['S1_SECONDS', 'S2_SECONDS', 'S3_SECONDS']].dropna()
            
            if len(speeds) > 0 and len(sector_times) > 0:
                # Identify slower sectors (likely more corners)
                avg_sector_times = sector_times.mean()
                slowest_sector = avg_sector_times.idxmax().replace('_SECONDS', '')
                
                # Average speed in slower sectors (corners)
                corner_speeds['avg_corner_speed'] = float(speeds.mean())
                corner_speeds['slowest_sector'] = slowest_sector
                corner_speeds['sector_speed_ratio'] = {
                    col.replace('_SECONDS', ''): float(avg_sector_times[col] / avg_sector_times.sum())
                    for col in sector_times.columns
                }
            else:
                corner_speeds['avg_corner_speed'] = 0
        else:
            corner_speeds['avg_corner_speed'] = 0
        
        return corner_speeds
    
    def _calculate_straight_corner_ratio(self, analysis_df: pd.DataFrame) -> Dict:
        """Calculate straight vs corner ratio from sector analysis."""
        sector_cols = [col for col in analysis_df.columns if col.startswith('S') and col.endswith('_SECONDS')]
        
        if len(sector_cols) < 2:
            return {'ratio': 0, 'method': 'insufficient_data'}
        
        sector_times = analysis_df[sector_cols].mean()
        
        # Heuristic: faster sectors = straights, slower sectors = corners
        avg_sector_time = sector_times.mean()
        
        # Count sectors significantly faster (straights) vs slower (corners)
        fast_threshold = avg_sector_time * 0.9
        slow_threshold = avg_sector_time * 1.1
        
        fast_sectors = (sector_times < fast_threshold).sum()
        slow_sectors = (sector_times > slow_threshold).sum()
        
        if slow_sectors > 0:
            ratio = fast_sectors / slow_sectors
        else:
            ratio = fast_sectors if fast_sectors > 0 else 1
        
        return {
            'ratio': float(ratio),
            'fast_sectors': int(fast_sectors),
            'slow_sectors': int(slow_sectors),
            'method': 'sector_time_analysis'
        }
    
    def _estimate_track_length(self, venue: str, race: str, analysis_df: pd.DataFrame) -> Dict:
        """Estimate track length from lap times and speeds."""
        if 'LAP_TIME_seconds' in analysis_df.columns and 'KPH' in analysis_df.columns:
            lap_times = analysis_df['LAP_TIME_seconds'].dropna()
            speeds = analysis_df['KPH'].dropna()
            
            if len(lap_times) > 0 and len(speeds) > 0:
                # Track length = average speed * average lap time
                avg_speed_ms = speeds.mean() / 3.6  # Convert km/h to m/s
                avg_lap_time = lap_times.mean()
                estimated_length = avg_speed_ms * avg_lap_time
                
                return {
                    'estimated_length_meters': float(estimated_length),
                    'estimated_length_km': float(estimated_length / 1000),
                    'method': 'speed_time_calculation',
                    'avg_speed_kmh': float(speeds.mean()),
                    'avg_lap_time_seconds': float(avg_lap_time)
                }
        
        venue_key = venue.lower()
        if venue_key in KNOWN_TRACK_LENGTHS_KM:
            km = KNOWN_TRACK_LENGTHS_KM[venue_key]
            return {
                'estimated_length_meters': float(km * 1000),
                'estimated_length_km': float(km),
                'method': 'known_reference'
            }
        
        return {'estimated_length_meters': 0, 'estimated_length_km': 0, 'method': 'insufficient_data'}
    
    def _extract_elevation_data(self, venue: str, race: str) -> Dict:
        """Extract elevation changes from telemetry if available."""
        try:
            telemetry = self._get_full_telemetry(venue, race)
            
            if telemetry.empty:
                return {'available': False, 'method': 'no_telemetry'}
            
            # Look for elevation-related parameters
            if 'parameter_name' in telemetry.columns:
                elevation_params = telemetry[telemetry['parameter_name'].str.contains('elev|altitude|height', case=False, na=False)]
                
                if not elevation_params.empty:
                    values = pd.to_numeric(elevation_params['value'], errors='coerce').dropna()
                    if len(values) > 0:
                        return {
                            'available': True,
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'range': float(values.max() - values.min()),
                            'mean': float(values.mean()),
                            'method': 'telemetry'
                        }
            
            return {'available': False, 'method': 'no_elevation_data'}
        except Exception as e:
            logger.warning(f"Error extracting elevation: {e}")
            return {'available': False, 'method': 'error', 'error': str(e)}

    def _get_full_telemetry(self, venue: str, race: str) -> pd.DataFrame:
        """
        Load and cache a large telemetry sample for the given venue/race.
        Keeps memory usage predictable while ensuring feature calculations see enough data.
        """
        key = (venue, race)
        if key not in self._telemetry_cache:
            telemetry_df = self.loader.load_telemetry(venue, race, sample_size=200000)
            if telemetry_df is None or telemetry_df.empty:
                self._telemetry_cache[key] = pd.DataFrame()
            else:
                # store a copy so downstream transformations do not mutate cached data
                self._telemetry_cache[key] = telemetry_df.copy()
        return self._telemetry_cache[key]
    
    def _calculate_sector_consistency(self, analysis_df: pd.DataFrame) -> Dict:
        """Calculate sector consistency metrics."""
        sector_cols = [col for col in analysis_df.columns if col.startswith('S') and col.endswith('_SECONDS')]
        
        if len(sector_cols) == 0:
            return {}
        
        consistency = {}
        
        for col in sector_cols:
            sector_data = analysis_df[col].dropna()
            if len(sector_data) > 0:
                sector_name = col.replace('_SECONDS', '')
                mean_time = sector_data.mean()
                std_time = sector_data.std()
                
                consistency[sector_name] = {
                    'cv': float(std_time / mean_time) if mean_time > 0 else 0,
                    'std': float(std_time),
                    'mean': float(mean_time),
                    'consistency_score': float(1 / (1 + std_time / mean_time)) if mean_time > 0 else 0
                }
        
        # Overall consistency
        if consistency:
            avg_cv = np.mean([v['cv'] for v in consistency.values()])
            consistency['overall'] = {
                'avg_coefficient_of_variation': float(avg_cv),
                'consistency_score': float(1 / (1 + avg_cv))
            }
        
        return consistency
    
    def _analyze_overtaking_opportunities(self, venue: str, race: str, 
                                         results_df: pd.DataFrame, 
                                         analysis_df: pd.DataFrame) -> Dict:
        """Analyze overtaking opportunities from position changes."""
        overtaking = {}
        
        # From results: analyze position changes
        if not results_df.empty and 'POSITION' in results_df.columns:
            # Simple heuristic: tracks with more position variance = more overtaking
            positions = results_df['POSITION'].dropna()
            if len(positions) > 0:
                overtaking['position_variance'] = float(positions.var())
                overtaking['position_range'] = float(positions.max() - positions.min())
        
        # From analysis: sector time differences (closer times = more overtaking opportunities)
        if 'LAP_TIME_seconds' in analysis_df.columns:
            lap_times = analysis_df['LAP_TIME_seconds'].dropna()
            if len(lap_times) > 0:
                # Calculate how close lap times are (smaller std = more competitive = more overtaking)
                time_std = lap_times.std()
                time_mean = lap_times.mean()
                overtaking['lap_time_competitiveness'] = {
                    'std': float(time_std),
                    'cv': float(time_std / time_mean) if time_mean > 0 else 0,
                    'overtaking_score': float(1 / (1 + time_std))  # Lower std = higher score
                }
        
        # Estimate from sector differences
        sector_cols = [col for col in analysis_df.columns if col.startswith('S') and col.endswith('_SECONDS')]
        if len(sector_cols) >= 2:
            # More variance in sector times = more opportunities for different strategies
            sector_variance = analysis_df[sector_cols].var().mean()
            overtaking['sector_variance'] = float(sector_variance)
            overtaking['strategy_diversity_score'] = float(sector_variance / 10)  # Normalized
        
        return overtaking
    
    def _calculate_complexity_score(self, complexity: Dict) -> float:
        """Calculate overall technical complexity score (0-1)."""
        score = 0.0
        weight_sum = 0.0

        # Factor 1: sector standard deviation (higher = more complex)
        sector_std = complexity.get('overall_sector_std')
        if sector_std is not None:
            # Typical challenging sector std ~0.8s
            var_score = float(np.clip(sector_std / 0.8, 0, 1))
            score += var_score * 0.35
            weight_sum += 0.35

        # Factor 2: braking zones (more zones or heavier braking = more complex)
        brake_count = complexity.get('braking_zones', {}).get('count')
        if brake_count is not None:
            brake_score = float(np.clip(brake_count / 12, 0, 1))
            score += brake_score * 0.25
            weight_sum += 0.25

        # Factor 3: corner speeds (lower average = tighter, more technical)
        corner_speed = complexity.get('corner_speeds', {}).get('avg_corner_speed')
        if corner_speed:
            corner_score = float(np.clip((180 - corner_speed) / 120, 0, 1))
            score += corner_score * 0.2
            weight_sum += 0.2

        # Factor 4: telemetry steering/acceleration activity
        telemetry = complexity.get('telemetry_features', {})
        steering_std = telemetry.get('steering_activity', 0)
        acceleration_var = telemetry.get('acceleration_variance', 0)
        if steering_std or acceleration_var:
            telemetry_score = float(
                np.clip((steering_std / 45.0) + (acceleration_var / 5.0), 0, 1)
            )
            score += telemetry_score * 0.2
            weight_sum += 0.2

        if weight_sum > 0:
            return score / weight_sum
        return 0.0
    
    # Empty dict helpers
    def _empty_complexity_dict(self) -> Dict:
        return {
            'sector_time_variance': {},
            'overall_sector_variance': 0,
            'braking_zones': {'count': 0},
            'corner_speeds': {},
            'complexity_score': 0.0
        }
    
    def _empty_speed_profile_dict(self) -> Dict:
        return {
            'top_speed': {},
            'speed_distribution': {},
            'straight_corner_ratio': {'ratio': 0}
        }
    
    def _empty_physical_dict(self) -> Dict:
        return {
            'track_length': {'estimated_length_meters': 0},
            'elevation': {'available': False},
            'num_sectors': 0,
            'sector_names': []
        }
    
    def _empty_performance_dict(self) -> Dict:
        return {
            'lap_time_variance': {},
            'sector_consistency': {},
            'overtaking_opportunities': {}
        }


def extract_all_tracks_dna(data_loader=None) -> pd.DataFrame:
    """
    Extract track DNA for all venues and races.
    
    Args:
        data_loader: DataLoader instance (optional)
        
    Returns:
        DataFrame with track DNA features for all tracks
    """
    if data_loader is None:
        from .data_loader import DataLoader
        loader = DataLoader()
    else:
        loader = data_loader
    
    extractor = TrackDNAExtractor(loader)
    
    all_dna = []
    
    for venue in loader.venues:
        for race in ["Race 1", "Race 2"]:
            try:
                dna = extractor.extract_track_dna(venue, race)
                all_dna.append(dna)
            except Exception as e:
                logger.error(f"Error extracting DNA for {venue} {race}: {e}")
    
    # Convert to DataFrame
    if all_dna:
        return pd.DataFrame(all_dna)
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    from .data_loader import DataLoader
    
    loader = DataLoader()
    extractor = TrackDNAExtractor(loader)
    
    # Extract DNA for a single track
    dna = extractor.extract_track_dna("barber", "Race 1")
    print("Track DNA for Barber Race 1:")
    print(f"Complexity Score: {dna['technical_complexity']['complexity_score']:.3f}")
    print(f"Track Length: {dna['physical_characteristics']['track_length']}")
    print(f"Number of Sectors: {dna['physical_characteristics']['num_sectors']}")

