"""
Track Clustering & Classification Module
Phase 1.2: Use Track DNA features to group similar tracks and create taxonomy

Features:
- Unsupervised clustering of tracks based on DNA features
- Track classification (technical, speed-focused, balanced)
- Track similarity matrix
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TrackClusterer:
    """
    Cluster and classify tracks based on Track DNA features.
    """
    
    def __init__(self, data_loader=None, track_dna_extractor=None):
        if data_loader is None:
            from .data_loader import DataLoader
            self.loader = DataLoader()
        else:
            self.loader = data_loader
            
        if track_dna_extractor is None:
            from .track_dna_extractor import TrackDNAExtractor, extract_all_tracks_dna
            self.extractor = TrackDNAExtractor(self.loader)
            self.extract_all_dna = extract_all_tracks_dna
        else:
            self.extractor = track_dna_extractor
            from .track_dna_extractor import extract_all_tracks_dna
            self.extract_all_dna = extract_all_tracks_dna
        
        self.scaler = StandardScaler()
        self.feature_names = []
        self.cluster_model = None
        self.cluster_labels = None
        self.track_features_df = None
    
    def extract_features_for_clustering(self, venues: Optional[List[str]] = None) -> pd.DataFrame:
        logger.info("Extracting Track DNA features for clustering...")
        all_dna_df = self.extract_all_dna(self.loader)
        
        if all_dna_df.empty:
            logger.warning("No track DNA data available")
            return pd.DataFrame()
        
        if venues:
            all_dna_df = all_dna_df[all_dna_df['venue'].isin(venues)]
        
        feature_rows = []
        
        for idx, row in all_dna_df.iterrows():
            features = {
                'venue': row['venue'],
                'race': row['race'],
                'track_id': f"{row['venue']}_{row['race']}"
            }
            
            tech = row.get('technical_complexity', {})
            if isinstance(tech, dict):
                features['complexity_score'] = tech.get('complexity_score', 0)
                features['sector_variance'] = tech.get('overall_sector_variance', 0)
                features['braking_zones'] = tech.get('braking_zones', {}).get('count', 0)
                features['avg_corner_speed'] = tech.get('corner_speeds', {}).get('avg_corner_speed', 0)
                
                telemetry = tech.get('telemetry_features', {})
                if isinstance(telemetry, dict):
                    features['accel_variance'] = telemetry.get('acceleration_variance', 0)
                    features['steering_activity'] = telemetry.get('steering_activity', 0)
                    features['gear_changes'] = telemetry.get('gear_changes', 0)
                    features['throttle_variance'] = telemetry.get('throttle_variance', 0)
            
            speed = row.get('speed_profile', {})
            if isinstance(speed, dict):
                top_speed = speed.get('top_speed', {})
                if isinstance(top_speed, dict):
                    features['top_speed_max'] = top_speed.get('max', 0)
                    features['top_speed_mean'] = top_speed.get('mean', 0)
                
                speed_dist = speed.get('speed_distribution', {})
                if isinstance(speed_dist, dict):
                    features['avg_speed'] = speed_dist.get('mean', 0)
                    features['speed_std'] = speed_dist.get('std', 0)
                
                straight_corner = speed.get('straight_corner_ratio', {})
                if isinstance(straight_corner, dict):
                    features['straight_corner_ratio'] = straight_corner.get('ratio', 0)
            
            physical = row.get('physical_characteristics', {})
            if isinstance(physical, dict):
                track_len = physical.get('track_length', {})
                if isinstance(track_len, dict):
                    features['track_length_km'] = track_len.get('estimated_length_km', 0)
                features['num_sectors'] = physical.get('num_sectors', 0)
                features['num_intermediate_points'] = physical.get('num_intermediate_points', 0)
            
            patterns = row.get('performance_patterns', {})
            if isinstance(patterns, dict):
                lap_var = patterns.get('lap_time_variance', {})
                if isinstance(lap_var, dict):
                    features['lap_time_cv'] = lap_var.get('cv', 0)
                    features['lap_time_std'] = lap_var.get('std', 0)
                
                overtaking = patterns.get('overtaking_opportunities', {})
                if isinstance(overtaking, dict):
                    features['overtaking_score'] = overtaking.get('lap_time_competitiveness', {}).get('overtaking_score', 0)
            
            feature_rows.append(features)
        
        features_df = pd.DataFrame(feature_rows)
        
        if features_df.empty:
            logger.warning("No features extracted")
            return pd.DataFrame()
        
        self.track_features_df = features_df
        
        logger.info(f"Extracted features for {len(features_df)} tracks")
        return features_df
    
    def prepare_features_for_clustering(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        exclude_cols = ['venue', 'race', 'track_id']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[feature_cols].copy()
        X = X.fillna(X.mean())
        
        self.feature_names = feature_cols
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, feature_cols
    
    def cluster_tracks(self, n_clusters: Optional[int] = None, method: str = 'kmeans', 
                     features_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if features_df is None:
            features_df = self.extract_features_for_clustering()
        
        if features_df.empty:
            logger.error("No features available for clustering")
            return pd.DataFrame()
        
        X_scaled, feature_names = self.prepare_features_for_clustering(features_df)
        
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X_scaled)
            logger.info(f"Auto-detected optimal clusters: {n_clusters}")
        
        if method == 'kmeans':
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.cluster_labels = self.cluster_model.fit_predict(X_scaled)
        elif method == 'dbscan':
            self.cluster_model = DBSCAN(eps=0.5, min_samples=2)
            self.cluster_labels = self.cluster_model.fit_predict(X_scaled)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        features_df['cluster'] = self.cluster_labels
        features_df['cluster_label'] = features_df['cluster'].apply(lambda x: f"Cluster_{x}")
        
        return features_df
    
    def _find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 5) -> int:
        if len(X) < 3:
            return 1
        
        max_clusters = min(max_clusters, len(X) - 1)
        if max_clusters < 2:
            return 2
        
        scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                scores.append((k, score))
        
        if scores:
            return max(scores, key=lambda x: x[1])[0]
        
        return 2
    
    def classify_tracks(self, features_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if features_df is None:
            features_df = self.cluster_tracks()
        
        if features_df.empty or 'cluster' not in features_df.columns:
            logger.error("No clustered data available for classification")
            return pd.DataFrame()
        
        classifications = []
        
        for idx, row in features_df.iterrows():
            technical_score = (
                row.get('complexity_score', 0) * 0.3 +
                row.get('sector_variance', 0) * 0.2 +
                row.get('braking_zones', 0) * 0.2 +
                row.get('steering_activity', 0) * 0.15 +
                row.get('gear_changes', 0) * 0.15
            )
            
            speed_score = (
                row.get('top_speed_max', 0) * 0.3 +
                row.get('avg_speed', 0) * 0.3 +
                row.get('straight_corner_ratio', 0) * 0.4
            )
            
            technical_score = min(technical_score / 10, 1.0)
            speed_score = min(speed_score / 200, 1.0)
            
            if technical_score > 0.6 and technical_score > speed_score * 1.2:
                classification = 'technical'
            elif speed_score > 0.6 and speed_score > technical_score * 1.2:
                classification = 'speed-focused'
            else:
                classification = 'balanced'
            
            classifications.append({
                'track_id': row['track_id'],
                'classification': classification,
                'technical_score': technical_score,
                'speed_score': speed_score
            })
        
        classification_df = pd.DataFrame(classifications)
        
        features_df = features_df.merge(classification_df, on='track_id', how='left')
        
        return features_df
    
    def build_similarity_matrix(self, features_df: Optional[pd.DataFrame] = None, metric: str = 'euclidean') -> pd.DataFrame:
        if features_df is None:
            features_df = self.extract_features_for_clustering()
        
        if features_df.empty:
            logger.error("No features available for similarity matrix")
            return pd.DataFrame()
        
        X_scaled, feature_names = self.prepare_features_for_clustering(features_df)
        
        distances = pdist(X_scaled, metric=metric)
        distance_matrix = squareform(distances)
        
        max_dist = distance_matrix.max()
        if max_dist > 0:
            similarity_matrix = 1 - (distance_matrix / max_dist)
        else:
            similarity_matrix = np.ones_like(distance_matrix)
        
        track_ids = features_df['track_id'].values
        
        return pd.DataFrame(similarity_matrix, index=track_ids, columns=track_ids)

    def get_cluster_summary(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for each cluster.
        
        Args:
            features_df: DataFrame with cluster assignments
            
        Returns:
            DataFrame with cluster summaries
        """
        if 'cluster' not in features_df.columns:
            logger.error("No cluster assignments found")
            return pd.DataFrame()
        
        clusters = sorted(features_df['cluster'].unique())
        
        # Basic numeric summary
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['cluster']
        summary_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        summary = features_df.groupby('cluster')[summary_cols].agg(['mean', 'std']).round(3)
        summary.index.name = 'cluster'
        
        # Add cluster sizes
        cluster_sizes = features_df.groupby('cluster').size()
        summary['size'] = cluster_sizes
        
        # Add classification distribution (fixed to avoid dtype issues)
        if 'classification' in features_df.columns:
            # Build classification distribution manually to avoid MultiIndex issues
            classification_dist = {}
            for cluster in clusters:
                cluster_data = features_df[features_df['cluster'] == cluster]
                if 'classification' in cluster_data.columns:
                    class_counts = cluster_data['classification'].value_counts().to_dict()
                    classification_dist[cluster] = class_counts
                else:
                    classification_dist[cluster] = {}
            
            # Convert to Series with proper dtype
            classification_series = pd.Series(classification_dist, dtype='object')
            summary['classifications'] = classification_series
        
        return summary


def cluster_all_tracks(data_loader=None, n_clusters: Optional[int] = None) -> pd.DataFrame:
    clusterer = TrackClusterer(data_loader=data_loader)
    clustered_df = clusterer.cluster_tracks(n_clusters=n_clusters)
    classified_df = clusterer.classify_tracks(clustered_df)
    return classified_df


def build_track_similarity_matrix(data_loader=None) -> pd.DataFrame:
    clusterer = TrackClusterer(data_loader=data_loader)
    return clusterer.build_similarity_matrix()
