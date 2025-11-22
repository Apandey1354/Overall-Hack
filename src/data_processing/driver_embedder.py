"""
driver_embedder.py

Driver Embedding Creation Module (rewritten)
Phase 2.1: Create driver skill vectors for transfer learning

Final skill vector (8 dims):
[Technical, High-Speed, Consistency, Weather,
 Tech-Track, Speed-Track, Balanced-Track, Finish-Rate]

All final values are guaranteed to be within [0.0, 1.0].
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_minmax_scale(values: List[float]) -> np.ndarray:
    """Min-max scale list -> numpy array in [0,1]. Handles constant/empty lists gracefully."""
    if not values:
        return np.array([], dtype=float)
    arr = np.array(values, dtype=float)
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr)
    arr = np.nan_to_num(arr, nan=0.0)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if mx <= mn:
        # constant array: map non-zero to 1.0, zeros to 0.0
        return np.where(arr > 0, 1.0, 0.0)
    return (arr - mn) / (mx - mn)


def _clip01(x: np.ndarray) -> np.ndarray:
    """Clip numpy array to [0,1] and replace infinities/nan."""
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(x, 0.0, 1.0)


class DriverEmbedder:
    """
    Create driver skill embeddings/vectors for transfer learning.
    """

    def __init__(self, data_loader: Optional[Any] = None, track_performance_analyzer: Optional[Any] = None):
        # Lazy imports to avoid circular dependency during tests
        if data_loader is None:
            from .data_loader import DataLoader  # adjust import path as appropriate
            self.loader = DataLoader()
        else:
            self.loader = data_loader

        if track_performance_analyzer is None:
            from .track_performance_analyzer import TrackPerformanceAnalyzer  # adjust import path as appropriate
            self.performance_analyzer = TrackPerformanceAnalyzer(data_loader=self.loader)
        else:
            self.performance_analyzer = track_performance_analyzer

        # Cache for all fastest-lap speeds across dataset (for percentile)
        self._all_speeds_cache: Optional[np.ndarray] = None

    # ------------------------
    # Public interface
    # ------------------------
    def create_driver_embedding(self, driver_number: int) -> Dict[str, Any]:
        """Create complete skill embedding for a single driver."""
        logger.debug("Creating embedding for driver #%s", driver_number)

        tech = self._calculate_technical_proficiency(driver_number)
        high_speed = self._calculate_high_speed_proficiency(driver_number)
        consistency = self._calculate_consistency_metrics(driver_number)
        weather = self._calculate_weather_adaptability(driver_number)
        track_strengths = self._calculate_track_specific_strengths(driver_number, high_speed_score=high_speed.get('score', 0.0))

        embedding = {
            'driver_number': int(driver_number),
            'technical_proficiency': tech,
            'high_speed_proficiency': high_speed,
            'consistency_metrics': consistency,
            'weather_adaptability': weather,
            'track_specific_strengths': track_strengths
        }

        embedding['overall_skill_vector'] = self._create_skill_vector(embedding)
        return embedding

    def create_all_driver_embeddings(self) -> pd.DataFrame:
        """Create embeddings for all drivers present in results files."""
        logger.info("Creating embeddings for all drivers...")

        # discover drivers
        driver_numbers = set()
        for venue in getattr(self.loader, "venues", []):
            for race in ["Race 1", "Race 2"]:
                results_df = self.loader.load_results_file(venue, race, "provisional")
                if results_df.empty:
                    results_df = self.loader.load_results_file(venue, race, "official")
                if not results_df.empty and 'NUMBER' in results_df.columns:
                    driver_numbers.update(results_df['NUMBER'].dropna().unique())

        driver_numbers = sorted([int(d) for d in driver_numbers if pd.notna(d)])

        records = []
        for dnum in driver_numbers:
            try:
                emb = self.create_driver_embedding(dnum)
                records.append({
                    'driver_number': dnum,
                    'skill_vector': emb['overall_skill_vector'],
                    'technical_proficiency': emb['technical_proficiency'].get('score', 0.0),
                    'high_speed_proficiency': emb['high_speed_proficiency'].get('score', 0.0),
                    'consistency_score': emb['consistency_metrics'].get('consistency_score', 0.0),
                    'weather_adaptability': emb['weather_adaptability'].get('adaptability_score', 0.0),
                    'best_track_type': emb['track_specific_strengths'].get('best_track_type'),
                    'strengths': ', '.join(emb['track_specific_strengths'].get('strengths', [])),
                    'full_embedding': emb
                })
            except Exception as e:
                logger.exception("Failed to create embedding for driver %s: %s", dnum, e)

        df = pd.DataFrame(records)

        if df.empty:
            logger.warning("No driver embeddings produced.")
            return df

        # Ensure skill vectors are numpy arrays and have shape (8,)
        skill_matrix = np.vstack([sv if isinstance(sv, np.ndarray) else np.zeros(8) for sv in df['skill_vector'].values])

        # Percentile ranking per-dimension (0..1)
        ranked_matrix = np.zeros_like(skill_matrix, dtype=float)
        n = skill_matrix.shape[0]
        for i in range(skill_matrix.shape[1]):
            vals = skill_matrix[:, i]
            # argsort then assign percentiles; ties handled by rank order
            order = np.argsort(vals)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(n) / max(1, n - 1)
            ranked_matrix[:, i] = ranks

        # Combine: 70% original, 30% percentile (preserves meaning + distinctiveness)
        final_matrix = _clip01(skill_matrix * 0.7 + ranked_matrix * 0.3)

        # Update DataFrame
        df['skill_vector'] = list(final_matrix)

        logger.info("Created embeddings for %d drivers", len(df))
        return df

    # ------------------------
    # Component calculators
    # ------------------------
    def _calculate_technical_proficiency(self, driver_number: int) -> Dict[str, Any]:
        """Calculate technical proficiency score (0..1) using performance analyzer."""
        try:
            driver_perf = self.performance_analyzer.analyze_driver_performance_by_track_type()
        except Exception:
            driver_perf = pd.DataFrame(columns=['driver_number', 'track_type', 'avg_position', 'best_position', 'races_count', 'performance_score'])

        row = driver_perf[
            (driver_perf['driver_number'] == driver_number) & (driver_perf['track_type'] == 'technical')
        ]
        if row.empty:
            return {'score': 0.0, 'avg_position': None, 'best_position': None, 'races_count': 0, 'method': 'no_data'}

        row = row.iloc[0]
        avg_pos = float(row.get('avg_position')) if pd.notna(row.get('avg_position')) else None
        # Map avg position into 0..1 where lower position -> higher score.
        if avg_pos and avg_pos > 0:
            score = max(0.0, min(1.0, 1.0 - (avg_pos - 1) / 19.0))
        else:
            score = 0.0

        return {
            'score': float(score),
            'avg_position': float(avg_pos) if avg_pos else None,
            'best_position': float(row.get('best_position')) if pd.notna(row.get('best_position')) else None,
            'races_count': int(row.get('races_count', 0)),
            'method': 'track_type_analysis'
        }

    def _calculate_high_speed_proficiency(self, driver_number: int) -> Dict[str, Any]:
        """Calculate high-speed proficiency using track-type analysis or fallback to fastest-lap speeds."""
        try:
            driver_perf = self.performance_analyzer.analyze_driver_performance_by_track_type()
        except Exception:
            driver_perf = pd.DataFrame(columns=['driver_number', 'track_type', 'avg_position', 'races_count', 'performance_score', 'avg_fastest_lap_speed'])

        row = driver_perf[
            (driver_perf['driver_number'] == driver_number) & (driver_perf['track_type'] == 'speed-focused')
        ]
        if not row.empty:
            row = row.iloc[0]
            avg_pos = float(row.get('avg_position')) if pd.notna(row.get('avg_position')) else None
            if avg_pos and avg_pos > 0:
                pos_component = max(0.0, min(1.0, 1.0 - (avg_pos - 1) / 19.0))
            else:
                pos_component = 0.0
            # prefer performance_score if provided, else pos_component
            perf_score = row.get('performance_score')
            base_score = float(perf_score) if pd.notna(perf_score) else pos_component
            return {
                'score': float(_clip01(np.array([base_score]))[0]),
                'avg_position': float(avg_pos) if avg_pos else None,
                'top_speed_avg': float(row.get('avg_fastest_lap_speed')) if pd.notna(row.get('avg_fastest_lap_speed')) else None,
                'races_count': int(row.get('races_count', 0)),
                'method': 'track_type_analysis'
            }

        # Fallback: compute from fastest-lap speeds across venues
        top_speeds = []
        positions = []
        for venue in getattr(self.loader, "venues", []):
            for race in ["Race 1", "Race 2"]:
                results_df = self.loader.load_results_file(venue, race, "provisional")
                if results_df.empty:
                    results_df = self.loader.load_results_file(venue, race, "official")
                if results_df.empty:
                    continue
                driver_result = results_df[results_df['NUMBER'] == driver_number]
                if driver_result.empty:
                    continue
                fl_speed = driver_result.iloc[0].get('FL_KPH', None)
                pos = driver_result.iloc[0].get('POSITION', None)
                if pd.notna(fl_speed):
                    top_speeds.append(float(fl_speed))
                if pd.notna(pos):
                    positions.append(float(pos))

        if top_speeds:
            avg_top_speed = float(np.mean(top_speeds))
            # expected speeds vary by series; convert to percentile vs dataset
            percentile = self._get_speed_percentile(avg_top_speed)
            # position component
            pos_component = 0.0
            if positions:
                avg_pos = float(np.mean(positions))
                pos_component = max(0.0, min(1.0, 1.0 - (avg_pos - 1) / 19.0))
            # combine: 70% percentile speed, 30% position (tunable)
            score = 0.7 * percentile + 0.3 * pos_component
            return {
                'score': float(_clip01(np.array([score]))[0]),
                'avg_position': float(np.mean(positions)) if positions else None,
                'top_speed_avg': avg_top_speed,
                'races_count': len(top_speeds),
                'method': 'top_speed_fallback'
            }

        # no data
        return {'score': 0.0, 'avg_position': None, 'top_speed_avg': None, 'races_count': 0, 'method': 'no_data'}

    def _calculate_consistency_metrics(self, driver_number: int) -> Dict[str, Any]:
        """Calculate consistency: position variance, lap time CV, finish rate, combined into 0..1 score."""
        consistency_data = []
        fastest_lap_times = []

        for venue in getattr(self.loader, "venues", []):
            for race in ["Race 1", "Race 2"]:
                results_df = self.loader.load_results_file(venue, race, "provisional")
                if results_df.empty:
                    results_df = self.loader.load_results_file(venue, race, "official")
                if results_df.empty:
                    continue
                driver_result = results_df[results_df['NUMBER'] == driver_number]
                if driver_result.empty:
                    continue

                pos = driver_result.iloc[0].get('POSITION', None)
                fl_time = driver_result.iloc[0].get('FL_TIME', None)
                status = driver_result.iloc[0].get('STATUS', '')
                finished = str(status).lower() not in ['dnf', 'dsq', 'nc', 'dnq']

                # parse M:SS.mmm or SS.mmm or mm:ss
                fl_seconds = None
                if isinstance(fl_time, str) and fl_time.strip():
                    try:
                        parts = fl_time.strip().split(':')
                        if len(parts) == 2:
                            minutes = int(parts[0])
                            seconds_parts = parts[1].split('.')
                            seconds = int(seconds_parts[0])
                            ms = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
                            fl_seconds = minutes * 60 + seconds + ms / 1000.0
                        else:
                            # maybe SS.mmm
                            s_parts = fl_time.split('.')
                            fl_seconds = float(s_parts[0]) + (float("0." + s_parts[1]) if len(s_parts) > 1 else 0.0)
                    except Exception:
                        fl_seconds = None

                if fl_seconds is not None:
                    fastest_lap_times.append(fl_seconds)

                consistency_data.append({
                    'position': float(pos) if pd.notna(pos) else None,
                    'finished': bool(finished),
                    'fastest_lap': fl_seconds
                })

        if not consistency_data:
            return {'lap_time_cv_avg': 0.0, 'position_variance': 0.0, 'finish_rate': 0.0, 'consistency_score': 0.0, 'races_analyzed': 0}

        df = pd.DataFrame(consistency_data)
        positions = df['position'].dropna()
        position_var = float(positions.var()) if len(positions) > 1 else 0.0

        fastest_laps = [x for x in fastest_lap_times if x is not None]
        if len(fastest_laps) > 1:
            arr = np.array(fastest_laps)
            lap_mean = arr.mean()
            lap_std = arr.std()
            lap_cv = float(lap_std / lap_mean) if lap_mean > 0 else 0.0
        else:
            lap_cv = 0.0

        finish_rate = float(df['finished'].mean()) if 'finished' in df.columns else 0.0

        # normalize components into roughly 0..1 ranges
        # lap_cv: typical 0.001..0.1 -> map with saturating transform
        cv_component = 1.0 / (1.0 + lap_cv * 100.0)  # maps small cv -> close to 1, large cv -> smaller
        cv_component = float(np.clip(cv_component, 0.0, 1.0))

        # position variance: expected range 0..100+, use 1/(1+var/10)
        pos_component = 1.0 / (1.0 + position_var / 10.0)
        pos_component = float(np.clip(pos_component, 0.0, 1.0))

        # combine with weights, clamp to [0,1]
        consistency_score = 0.5 * cv_component + 0.3 * pos_component + 0.2 * finish_rate
        consistency_score = float(np.clip(consistency_score, 0.0, 1.0))

        return {
            'lap_time_cv_avg': lap_cv,
            'position_variance': position_var,
            'finish_rate': float(np.clip(finish_rate, 0.0, 1.0)),
            'consistency_score': consistency_score,
            'races_analyzed': len(consistency_data)
        }

    def _calculate_weather_adaptability(self, driver_number: int) -> Dict[str, Any]:
        """Compute how similarly a driver performs in rain vs dry; bounded to [0,1]."""
        weather_perf = []

        for venue in getattr(self.loader, "venues", []):
            for race in ["Race 1", "Race 2"]:
                weather_df = self.loader.load_weather_data(venue, race)
                if weather_df.empty:
                    continue
                avg_temp = float(weather_df['AIR_TEMP'].mean()) if 'AIR_TEMP' in weather_df.columns and not weather_df['AIR_TEMP'].isna().all() else None
                has_rain = bool(weather_df['RAIN'].sum() > 0) if 'RAIN' in weather_df.columns else False

                results_df = self.loader.load_results_file(venue, race, "provisional")
                if results_df.empty:
                    results_df = self.loader.load_results_file(venue, race, "official")
                if results_df.empty:
                    continue

                driver_result = results_df[results_df['NUMBER'] == driver_number]
                if driver_result.empty:
                    continue

                pos = driver_result.iloc[0].get('POSITION', None)
                status = driver_result.iloc[0].get('STATUS', '')
                finished = str(status).lower() not in ['dnf', 'dsq', 'nc', 'dnq']

                if pd.notna(pos):
                    weather_perf.append({
                        'temperature': avg_temp,
                        'has_rain': bool(has_rain),
                        'position': float(pos),
                        'finished': bool(finished)
                    })

        if not weather_perf:
            return {'adaptability_score': 0.0, 'rain_performance': None, 'dry_performance': None, 'temperature_range': None, 'conditions_raced': 0}

        wdf = pd.DataFrame(weather_perf)
        rain_df = wdf[wdf['has_rain'] == True]
        dry_df = wdf[wdf['has_rain'] == False]

        rain_perf = None
        if not rain_df.empty:
            rain_avg_pos = float(rain_df['position'].mean())
            rain_perf = (1.0 / rain_avg_pos) if rain_avg_pos > 0 else 0.0

        dry_perf = None
        if not dry_df.empty:
            dry_avg_pos = float(dry_df['position'].mean())
            dry_perf = (1.0 / dry_avg_pos) if dry_avg_pos > 0 else 0.0

        # Bounded adaptability: if both exist, reward closeness, clamp difference to [0,1]
        if (rain_perf is not None) and (dry_perf is not None):
            diff = abs(rain_perf - dry_perf)
            diff = min(diff, 1.0)
            adaptability_score = 1.0 - diff
        elif dry_perf is not None:
            adaptability_score = float(np.clip(dry_perf, 0.0, 1.0))
        elif rain_perf is not None:
            adaptability_score = float(np.clip(rain_perf, 0.0, 1.0))
        else:
            adaptability_score = 0.0

        temp_range = None
        if 'temperature' in wdf.columns and wdf['temperature'].notna().any():
            temp_range = {
                'min': float(wdf['temperature'].min()),
                'max': float(wdf['temperature'].max()),
                'mean': float(wdf['temperature'].mean())
            }

        return {
            'adaptability_score': float(np.clip(adaptability_score, 0.0, 1.0)),
            'rain_performance': float(np.clip(rain_perf, 0.0, 1.0)) if rain_perf is not None else None,
            'dry_performance': float(np.clip(dry_perf, 0.0, 1.0)) if dry_perf is not None else None,
            'temperature_range': temp_range,
            'conditions_raced': len(weather_perf)
        }

    def _calculate_track_specific_strengths(self, driver_number: int, high_speed_score: float = 0.0) -> Dict[str, Any]:
        """
        Calculate track-specific strengths and weaknesses.
        Returns per-type raw scores, normalized track_type_scores that sum to 1 (if any non-zero).
        """
        try:
            driver_perf = self.performance_analyzer.analyze_driver_performance_by_track_type()
        except Exception:
            driver_perf = pd.DataFrame(columns=['driver_number', 'track_type', 'performance_score', 'races_count'])

        driver_data = driver_perf[driver_perf['driver_number'] == driver_number]

        raw = {'technical': 0.0, 'speed-focused': 0.0, 'balanced': 0.0}
        for t in raw.keys():
            row = driver_data[driver_data['track_type'] == t]
            if not row.empty:
                perf = row.iloc[0].get('performance_score', None)
                if pd.notna(perf):
                    raw[t] = float(perf)
                else:
                    # fallback to avg position if available
                    avg_pos = row.iloc[0].get('avg_position', None)
                    if pd.notna(avg_pos) and float(avg_pos) > 0:
                        raw[t] = float(max(0.0, min(1.0, 1.0 - (float(avg_pos) - 1) / 19.0)))
            else:
                raw[t] = 0.0

        # If speed-focused zero, attempt to compute via fastest-lap percentile or high_speed_score
        if raw['speed-focused'] == 0.0:
            # gather avg fastest-lap speed for this driver across events
            top_speeds = []
            for venue in getattr(self.loader, "venues", []):
                for race in ["Race 1", "Race 2"]:
                    results_df = self.loader.load_results_file(venue, race, "provisional")
                    if results_df.empty:
                        results_df = self.loader.load_results_file(venue, race, "official")
                    if results_df.empty or 'FL_KPH' not in results_df.columns:
                        continue
                    driver_result = results_df[results_df['NUMBER'] == driver_number]
                    if driver_result.empty:
                        continue
                    fl = driver_result.iloc[0].get('FL_KPH', None)
                    if pd.notna(fl):
                        top_speeds.append(float(fl))
            if top_speeds:
                avg_speed = float(np.mean(top_speeds))
                speed_pct = self._get_speed_percentile(avg_speed)
                raw['speed-focused'] = max(raw['speed-focused'], speed_pct, float(high_speed_score))

            else:
                raw['speed-focused'] = max(raw['speed-focused'], float(high_speed_score))

        # Now raw values exist; convert to normalized distribution that sums to 1 (if there is signal)
        raw_vals = np.array([raw['technical'], raw['speed-focused'], raw['balanced']], dtype=float)
        # if all zeros -> fallback tiny, deterministic variation to avoid identical vectors
        if np.allclose(raw_vals, 0.0):
            driver_num = int(driver_number) % 100
            fallback = np.array([
                0.001 + (driver_num % 10) / 10000.0,
                0.001 + ((driver_num // 10) % 10) / 10000.0,
                0.001 + ((driver_num // 100) % 10) / 10000.0
            ], dtype=float)
            raw_vals = fallback

        # If any values negative (shouldn't happen) clip
        raw_vals = np.clip(raw_vals, 0.0, None)
        if raw_vals.sum() > 0:
            track_scores = raw_vals / raw_vals.sum()
        else:
            # improbable
            track_scores = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)

        track_type_scores = {
            'technical': float(track_scores[0]),
            'speed-focused': float(track_scores[1]),
            'balanced': float(track_scores[2])
        }

        avg_score = float(track_scores.mean()) if track_scores.size > 0 else 0.0
        strengths = [k for k, v in track_type_scores.items() if v > avg_score * 1.1]
        weaknesses = [k for k, v in track_type_scores.items() if v < avg_score * 0.9]
        best_track_type = max(track_type_scores.items(), key=lambda x: x[1])[0]

        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'track_type_scores': track_type_scores,
            'best_track_type': best_track_type,
            'average_score': float(avg_score)
        }

    # ------------------------
    # Utility helpers
    # ------------------------
    def _get_speed_percentile(self, speed_value: float) -> float:
        """Return percentile (0..1) of a given FL_KPH relative to all cached FL_KPH values."""
        if speed_value is None:
            return 0.0
        if self._all_speeds_cache is None:
            all_speeds = []
            for venue in getattr(self.loader, "venues", []):
                for race in ["Race 1", "Race 2"]:
                    results_df = self.loader.load_results_file(venue, race, "provisional")
                    if results_df.empty:
                        results_df = self.loader.load_results_file(venue, race, "official")
                    if results_df.empty or 'FL_KPH' not in results_df.columns:
                        continue
                    speeds = results_df['FL_KPH'].dropna().astype(float).tolist()
                    all_speeds.extend(speeds)
            if all_speeds:
                self._all_speeds_cache = np.array(all_speeds, dtype=float)
            else:
                self._all_speeds_cache = np.array([], dtype=float)

        if self._all_speeds_cache.size == 0:
            return 0.0

        # percentile: fraction of speeds strictly less than given value
        pct = float(np.sum(self._all_speeds_cache < speed_value) / self._all_speeds_cache.size)
        return float(np.clip(pct, 0.0, 1.0))

    def _create_skill_vector(self, embedding: Dict[str, Any]) -> np.ndarray:
        """
        Compose the 8-dimensional vector and ensure strict 0..1 bounds.
        Order: [tech, speed, consistency, weather, techTrack, speedTrack, balancedTrack, finishRate]
        """
        tech_score = float(embedding.get('technical_proficiency', {}).get('score', 0.0))
        speed_score = float(embedding.get('high_speed_proficiency', {}).get('score', 0.0))
        consistency_score = float(embedding.get('consistency_metrics', {}).get('consistency_score', 0.0))
        weather_score = float(embedding.get('weather_adaptability', {}).get('adaptability_score', 0.0))

        track_scores = embedding.get('track_specific_strengths', {}).get('track_type_scores', {})
        tech_track = float(track_scores.get('technical', 0.0))
        speed_track = float(track_scores.get('speed-focused', 0.0))
        balanced_track = float(track_scores.get('balanced', 0.0))

        finish_rate = float(embedding.get('consistency_metrics', {}).get('finish_rate', 0.0))

        # If some track-score is near-zero, apply a tiny deterministic variation based on driver number
        driver_num = int(embedding.get('driver_number', 0))
        base_variation = (driver_num % 100) / 10000.0  # range 0.00-0.0099

        if tech_track < 0.001:
            tech_track = base_variation + tech_track
        if speed_track < 0.001:
            speed_track = base_variation + speed_track
        if balanced_track < 0.001:
            balanced_track = base_variation + balanced_track

        # Compose vector
        vec = np.array([
            tech_score,
            speed_score,
            consistency_score,
            weather_score,
            tech_track,
            speed_track,
            balanced_track,
            finish_rate
        ], dtype=float)

        # Final robust normalization:
        # - NaN/infs -> safe numbers, then clip to 0..1
        vec = _clip01(vec)

        return vec.astype(np.float32)


# convenience function
def create_driver_embeddings(data_loader: Optional[Any] = None) -> pd.DataFrame:
    embedder = DriverEmbedder(data_loader=data_loader)
    return embedder.create_all_driver_embeddings()
