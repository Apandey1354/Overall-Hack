"""
Track-Specific Performance Analysis Module
Phase 1.3: Analyze driver performance by track type and create difficulty rankings

Features:
- Driver performance analysis by track classification
- Track-specific driving pattern identification
- Track difficulty rankings
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class TrackPerformanceAnalyzer:
    """
    Analyze driver performance by track type and create difficulty rankings.
    """
    
    def __init__(self, data_loader=None, track_clusterer=None):
        """
        Initialize the Track Performance Analyzer.
        
        Args:
            data_loader: DataLoader instance (optional)
            track_clusterer: TrackClusterer instance (optional)
        """
        if data_loader is None:
            from .data_loader import DataLoader
            self.loader = DataLoader()
        else:
            self.loader = data_loader
            
        if track_clusterer is None:
            from .track_clustering import TrackClusterer, cluster_all_tracks
            self.clusterer = TrackClusterer(data_loader=self.loader)
            self.clustered_tracks = None
        else:
            self.clusterer = track_clusterer
            self.clustered_tracks = None
    
    def get_track_classifications(self) -> pd.DataFrame:
        """
        Get track classifications (cluster and type).
        
        Returns:
            DataFrame with track classifications
        """
        if self.clustered_tracks is None:
            self.clustered_tracks = self.clusterer.cluster_tracks()
            self.clustered_tracks = self.clusterer.classify_tracks(self.clustered_tracks)
        
        return self.clustered_tracks[['track_id', 'venue', 'race', 'cluster', 'classification']].copy()
    
    def analyze_driver_performance_by_track_type(self, venues: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze which drivers excel at which track types.
        
        Args:
            venues: List of venues to analyze (None = all)
            
        Returns:
            DataFrame with driver performance by track type
        """
        logger.info("Analyzing driver performance by track type...")
        
        # Get track classifications
        track_classifications = self.get_track_classifications()
        
        # Get all results data
        driver_performance = []
        
        for venue in (venues or self.loader.venues):
            for race in ["Race 1", "Race 2"]:
                # Get track classification
                track_id = f"{venue}_{race}"
                track_info = track_classifications[track_classifications['track_id'] == track_id]
                
                if track_info.empty:
                    continue
                
                track_class = track_info.iloc[0]['classification']
                track_cluster = track_info.iloc[0]['cluster']
                
                # Load results
                results_df = self.loader.load_results_file(venue, race, "provisional")
                if results_df.empty:
                    results_df = self.loader.load_results_file(venue, race, "official")
                
                if results_df.empty:
                    continue
                
                # Extract driver performance metrics
                for idx, row in results_df.iterrows():
                    driver_num = row.get('NUMBER', None)
                    if pd.isna(driver_num):
                        continue
                    
                    # Performance metrics
                    position = row.get('POSITION', None)
                    laps = row.get('LAPS', None)
                    fastest_lap_time = row.get('FL_TIME', None)
                    fastest_lap_speed = row.get('FL_KPH', None)
                    
                    # Convert fastest lap time to seconds if available
                    if fastest_lap_time and isinstance(fastest_lap_time, str):
                        try:
                            # Parse M:SS.mmm format
                            parts = fastest_lap_time.split(':')
                            if len(parts) == 2:
                                minutes = int(parts[0])
                                seconds_parts = parts[1].split('.')
                                seconds = int(seconds_parts[0])
                                milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
                                fastest_lap_seconds = minutes * 60 + seconds + milliseconds / 1000
                            else:
                                fastest_lap_seconds = None
                        except:
                            fastest_lap_seconds = None
                    else:
                        fastest_lap_seconds = None
                    
                    driver_performance.append({
                        'driver_number': int(driver_num),
                        'venue': venue,
                        'race': race,
                        'track_id': track_id,
                        'track_classification': track_class,
                        'track_cluster': track_cluster,
                        'position': int(position) if pd.notna(position) else None,
                        'laps_completed': int(laps) if pd.notna(laps) else None,
                        'fastest_lap_time': fastest_lap_seconds,
                        'fastest_lap_speed': float(fastest_lap_speed) if pd.notna(fastest_lap_speed) else None,
                        'finished': row.get('STATUS', '').lower() not in ['dnf', 'dsq', 'nc']
                    })
        
        performance_df = pd.DataFrame(driver_performance)
        
        if performance_df.empty:
            logger.warning("No driver performance data found")
            return pd.DataFrame()
        
        # Calculate performance scores by track type
        performance_by_type = []
        
        for driver_num in performance_df['driver_number'].unique():
            driver_data = performance_df[performance_df['driver_number'] == driver_num]
            
            for track_type in ['technical', 'speed-focused', 'balanced']:
                type_data = driver_data[driver_data['track_classification'] == track_type]
                
                if len(type_data) == 0:
                    continue
                
                # Calculate metrics
                avg_position = type_data['position'].mean() if type_data['position'].notna().any() else None
                best_position = type_data['position'].min() if type_data['position'].notna().any() else None
                avg_fastest_lap = type_data['fastest_lap_time'].mean() if type_data['fastest_lap_time'].notna().any() else None
                best_fastest_lap = type_data['fastest_lap_time'].min() if type_data['fastest_lap_time'].notna().any() else None
                avg_speed = type_data['fastest_lap_speed'].mean() if type_data['fastest_lap_speed'].notna().any() else None
                finish_rate = type_data['finished'].sum() / len(type_data) if len(type_data) > 0 else 0
                
                # Performance score (lower position = better, lower lap time = better)
                if avg_position is not None:
                    # Invert position (1st = best, higher = worse)
                    position_score = 1 / avg_position if avg_position > 0 else 0
                else:
                    position_score = 0
                
                performance_by_type.append({
                    'driver_number': driver_num,
                    'track_type': track_type,
                    'races_count': len(type_data),
                    'avg_position': avg_position,
                    'best_position': best_position,
                    'avg_fastest_lap_time': avg_fastest_lap,
                    'best_fastest_lap_time': best_fastest_lap,
                    'avg_fastest_lap_speed': avg_speed,
                    'finish_rate': finish_rate,
                    'performance_score': position_score,
                    'tracks': ', '.join(type_data['track_id'].unique().tolist())
                })
        
        result_df = pd.DataFrame(performance_by_type)
        
        logger.info(f"Analyzed performance for {len(result_df)} driver-track_type combinations")
        return result_df
    
    def identify_track_specific_patterns(self, venue: str, race: str) -> Dict:
        """
        Identify track-specific driving patterns from telemetry and lap data.
        
        Args:
            venue: Track venue name
            race: Race number
            
        Returns:
            Dictionary with identified patterns
        """
        logger.info(f"Identifying patterns for {venue} - {race}")
        
        patterns = {
            'venue': venue,
            'race': race,
            'track_id': f"{venue}_{race}"
        }
        
        # Load analysis data
        analysis_df = self.loader.load_analysis_endurance(venue, race)
        
        if analysis_df.empty:
            logger.warning(f"No analysis data for {venue} - {race}")
            return patterns
        
        # Pattern 1: Sector consistency patterns
        sector_cols = [col for col in analysis_df.columns if col.startswith('S') and col.endswith('_SECONDS')]
        if len(sector_cols) >= 2:
            sector_means = analysis_df[sector_cols].mean()
            sector_stds = analysis_df[sector_cols].std()
            
            # Identify most variable sector (where drivers struggle most)
            most_variable_sector = sector_stds.idxmax().replace('_SECONDS', '')
            least_variable_sector = sector_stds.idxmin().replace('_SECONDS', '')
            
            patterns['sector_patterns'] = {
                'most_variable_sector': most_variable_sector,
                'least_variable_sector': least_variable_sector,
                'sector_variance_ratio': float(sector_stds.max() / sector_stds.min()) if sector_stds.min() > 0 else 0
            }
        
        # Pattern 2: Speed distribution patterns
        if 'KPH' in analysis_df.columns:
            speeds = analysis_df['KPH'].dropna()
            if len(speeds) > 0:
                patterns['speed_patterns'] = {
                    'speed_range': float(speeds.max() - speeds.min()),
                    'speed_consistency': float(1 / (1 + speeds.std())) if speeds.std() > 0 else 0,
                    'high_speed_laps_ratio': float((speeds > speeds.quantile(0.75)).sum() / len(speeds))
                }
        
        # Pattern 3: Lap time progression (improvement over race)
        if 'LAP_TIME_seconds' in analysis_df.columns:
            lap_times = analysis_df['LAP_TIME_seconds'].dropna()
            if len(lap_times) > 10:
                # Check if drivers improve over time
                first_half = lap_times[:len(lap_times)//2].mean()
                second_half = lap_times[len(lap_times)//2:].mean()
                improvement = first_half - second_half  # Positive = improvement
                
                patterns['lap_progression'] = {
                    'improvement_over_race': float(improvement),
                    'early_race_avg': float(first_half),
                    'late_race_avg': float(second_half),
                    'shows_improvement': improvement > 0
                }
        
        # Pattern 4: Position change patterns (from results)
        results_df = self.loader.load_results_file(venue, race, "provisional")
        if results_df.empty:
            results_df = self.loader.load_results_file(venue, race, "official")
        
        if not results_df.empty and 'POSITION' in results_df.columns:
            positions = results_df['POSITION'].dropna()
            if len(positions) > 0:
                patterns['competitiveness'] = {
                    'position_spread': float(positions.max() - positions.min()),
                    'position_variance': float(positions.var()),
                    'close_competition': float(positions.std() < positions.mean() * 0.3)  # Tight field
                }
        
        return patterns
    
    def create_track_difficulty_rankings(self) -> pd.DataFrame:
        """
        Create track difficulty rankings based on multiple factors.
        
        Returns:
            DataFrame with track difficulty scores and rankings
        """
        logger.info("Creating track difficulty rankings...")
        
        # Get track classifications
        track_classifications = self.get_track_classifications()
        
        difficulty_scores = []
        
        for idx, row in track_classifications.iterrows():
            venue = row['venue']
            race = row['race']
            track_id = row['track_id']
            
            # Load data for difficulty metrics
            results_df = self.loader.load_results_file(venue, race, "provisional")
            if results_df.empty:
                results_df = self.loader.load_results_file(venue, race, "official")
            
            analysis_df = self.loader.load_analysis_endurance(venue, race)
            
            difficulty_factors = {}
            
            # Factor 1: Position variance (higher = more difficult/unpredictable)
            if not results_df.empty and 'POSITION' in results_df.columns:
                positions = results_df['POSITION'].dropna()
                if len(positions) > 0:
                    difficulty_factors['position_variance'] = float(positions.var())
                    difficulty_factors['position_range'] = float(positions.max() - positions.min())
            
            # Factor 2: Lap time variance (higher = more difficult)
            if 'LAP_TIME_seconds' in analysis_df.columns:
                lap_times = analysis_df['LAP_TIME_seconds'].dropna()
                if len(lap_times) > 0:
                    difficulty_factors['lap_time_variance'] = float(lap_times.var())
                    difficulty_factors['lap_time_cv'] = float(lap_times.std() / lap_times.mean())
            
            # Factor 3: Sector variance (higher = more technical/difficult)
            sector_cols = [col for col in analysis_df.columns if col.startswith('S') and col.endswith('_SECONDS')]
            if len(sector_cols) > 0:
                sector_variance = analysis_df[sector_cols].var().mean()
                difficulty_factors['sector_variance'] = float(sector_variance)
            
            # Factor 4: DNF rate (higher = more difficult)
            if not results_df.empty and 'STATUS' in results_df.columns:
                dnf_count = results_df['STATUS'].str.lower().isin(['dnf', 'dsq', 'nc']).sum()
                total_drivers = len(results_df)
                difficulty_factors['dnf_rate'] = float(dnf_count / total_drivers) if total_drivers > 0 else 0
            
            # Factor 5: Speed range (higher = more challenging)
            if 'KPH' in analysis_df.columns:
                speeds = analysis_df['KPH'].dropna()
                if len(speeds) > 0:
                    difficulty_factors['speed_range'] = float(speeds.max() - speeds.min())
            
            # Calculate overall difficulty score
            # Normalize and weight factors
            difficulty_score = (
                (difficulty_factors.get('lap_time_cv', 0) * 0.3) +
                (difficulty_factors.get('sector_variance', 0) / 10 * 0.25) +
                (difficulty_factors.get('dnf_rate', 0) * 0.25) +
                (difficulty_factors.get('position_variance', 0) / 100 * 0.2)
            )
            
            difficulty_scores.append({
                'track_id': track_id,
                'venue': venue,
                'race': race,
                'track_classification': row['classification'],
                'track_cluster': row['cluster'],
                'difficulty_score': difficulty_score,
                **difficulty_factors
            })
        
        difficulty_df = pd.DataFrame(difficulty_scores)
        
        if not difficulty_df.empty:
            # Rank tracks by difficulty
            difficulty_df = difficulty_df.sort_values('difficulty_score', ascending=False)
            difficulty_df['difficulty_rank'] = range(1, len(difficulty_df) + 1)
            difficulty_df = difficulty_df.reset_index(drop=True)
        
        logger.info(f"Created difficulty rankings for {len(difficulty_df)} tracks")
        return difficulty_df
    
    def generate_performance_report(self, output_path: Optional[str] = None) -> Dict:
        """
        Generate comprehensive performance analysis report.
        
        Args:
            output_path: Path to save report (optional)
            
        Returns:
            Dictionary with report data
        """
        logger.info("Generating performance analysis report...")
        
        report = {
            'driver_performance_by_type': self.analyze_driver_performance_by_track_type(),
            'track_difficulty_rankings': self.create_track_difficulty_rankings(),
            'track_patterns': {}
        }
        
        # Get track-specific patterns for all tracks
        track_classifications = self.get_track_classifications()
        for idx, row in track_classifications.iterrows():
            patterns = self.identify_track_specific_patterns(row['venue'], row['race'])
            report['track_patterns'][row['track_id']] = patterns
        
        # Summary statistics
        if not report['driver_performance_by_type'].empty:
            report['summary'] = {
                'total_drivers_analyzed': report['driver_performance_by_type']['driver_number'].nunique(),
                'drivers_by_track_type': report['driver_performance_by_type'].groupby('track_type')['driver_number'].nunique().to_dict(),
                'top_performers': {}
            }
            
            # Top performers by track type
            for track_type in ['technical', 'speed-focused', 'balanced']:
                type_data = report['driver_performance_by_type'][
                    report['driver_performance_by_type']['track_type'] == track_type
                ]
                if not type_data.empty:
                    top = type_data.nlargest(5, 'performance_score')
                    report['summary']['top_performers'][track_type] = top[['driver_number', 'performance_score', 'avg_position']].to_dict('records')
        
        # Save report if path provided
        if output_path:
            import json
            # Convert DataFrames to dict for JSON serialization
            report_dict = {
                'summary': report['summary'],
                'track_difficulty_rankings': report['track_difficulty_rankings'].to_dict('records'),
                'track_patterns': report['track_patterns']
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            logger.info(f"Report saved to {output_path}")
        
        return report


def analyze_track_performance(data_loader=None, output_path: Optional[str] = None) -> Dict:
    """
    Convenience function to run complete track performance analysis.
    
    Args:
        data_loader: DataLoader instance (optional)
        output_path: Path to save report (optional)
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = TrackPerformanceAnalyzer(data_loader=data_loader)
    return analyzer.generate_performance_report(output_path=output_path)

