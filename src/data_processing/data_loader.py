"""
Unified Data Loader for GR Cup Racing Data
Handles all CSV file types across all venues
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loader for GR Cup racing datasets.
    Handles multiple file formats, venues, and data types.
    """
    
    def __init__(self, base_path: str = ".", config_path: str = None):
        """
        Initialize the data loader.
        
        Args:
            base_path: Base directory containing the data
            config_path: Path to configuration file (if None, will search for config.yaml)
        """
        self.base_path = Path(base_path).resolve()
        
        # If config_path not provided, try to find it relative to base_path or project root
        if config_path is None:
            # Try relative to base_path first
            config_path = self.base_path / "config" / "config.yaml"
            if not config_path.exists():
                # Try relative to current file location (project root)
                project_root = Path(__file__).parent.parent.parent
                config_path = project_root / "config" / "config.yaml"
        else:
            config_path = Path(config_path)
            if not config_path.is_absolute():
                # Make relative to base_path
                config_path = self.base_path / config_path
        
        self.config = self._load_config(str(config_path))
        self.venues = self.config.get('data', {}).get('venues', [])
        self.file_patterns = self.config.get('data', {}).get('file_patterns', {})
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return {}
    
    def _detect_delimiter(self, file_path: Path) -> str:
        """
        Detect CSV delimiter (semicolon or comma).
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Detected delimiter
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if ';' in first_line:
                return ';'
            return ','
    
    def _parse_time_string(self, time_str: str) -> Optional[float]:
        """
        Parse time string (MM:SS.mmm or M:SS.mmm) to seconds.
        
        Args:
            time_str: Time string to parse
            
        Returns:
            Time in seconds or None if invalid
        """
        if pd.isna(time_str) or time_str == '' or time_str == '-':
            return None
        
        try:
            parts = str(time_str).split(':')
            if len(parts) == 2:
                minutes, seconds = parts
                return float(minutes) * 60 + float(seconds)
            elif len(parts) == 3:
                hours, minutes, seconds = parts
                return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
            else:
                return float(time_str)
        except (ValueError, AttributeError):
            return None
    
    def load_results_file(self, venue: str, race: str, file_type: str = "provisional") -> pd.DataFrame:
        """
        Load race results file.
        
        Args:
            venue: Track venue name
            race: Race number ("Race 1" or "Race 2")
            file_type: "provisional" or "official"
            
        Returns:
            DataFrame with race results
        """
        venue_path = self._get_venue_path(venue)
        
        if file_type == "official":
            pattern = f"**/*Results*{race}*Official*.CSV"
        else:
            pattern = f"**/*Provisional Results*{race}*.CSV"
        
        files = list(venue_path.glob(pattern))
        
        if not files:
            logger.warning(f"No results file found for {venue}, {race}, {file_type}")
            return pd.DataFrame()
        
        file_path = files[0]
        delimiter = self._detect_delimiter(file_path)
        
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
            df['venue'] = venue
            df['race'] = race
            df['file_type'] = file_type
            
            # Parse time columns
            time_cols = ['TOTAL_TIME', 'FL_TIME']
            for col in time_cols:
                if col in df.columns:
                    df[f'{col}_seconds'] = df[col].apply(self._parse_time_string)
            
            logger.info(f"Loaded results: {len(df)} rows from {file_path.name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading results file {file_path}: {e}")
            return pd.DataFrame()
    
    def load_results_by_class(self, venue: str, race: str, file_type: str = "provisional") -> pd.DataFrame:
        """
        Load results by class file.
        
        Args:
            venue: Track venue name
            race: Race number ("Race 1" or "Race 2")
            file_type: "provisional" or "official"
            
        Returns:
            DataFrame with results by class
        """
        venue_path = self._get_venue_path(venue)
        
        if file_type == "official":
            pattern = f"**/*Results by Class*{race}*Official*.CSV"
        else:
            pattern = f"**/*Results by Class*{race}*.CSV"
        
        files = list(venue_path.glob(pattern))
        
        if not files:
            logger.warning(f"No results by class file found for {venue}, {race}, {file_type}")
            return pd.DataFrame()
        
        file_path = files[0]
        delimiter = self._detect_delimiter(file_path)
        
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
            df['venue'] = venue
            df['race'] = race
            df['file_type'] = file_type
            
            # Parse time columns
            if 'ELAPSED' in df.columns:
                df['ELAPSED_seconds'] = df['ELAPSED'].apply(self._parse_time_string)
            if 'BEST_LAP_TIME' in df.columns:
                df['BEST_LAP_TIME_seconds'] = df['BEST_LAP_TIME'].apply(self._parse_time_string)
            
            logger.info(f"Loaded results by class: {len(df)} rows from {file_path.name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading results by class file {file_path}: {e}")
            return pd.DataFrame()
    
    def load_weather_data(self, venue: str, race: str) -> pd.DataFrame:
        """
        Load weather data file.
        
        Args:
            venue: Track venue name
            race: Race number ("Race 1" or "Race 2")
            
        Returns:
            DataFrame with weather data
        """
        venue_path = self._get_venue_path(venue)
        pattern = f"**/*Weather*{race}*.CSV"
        files = list(venue_path.glob(pattern))
        
        if not files:
            logger.warning(f"No weather file found for {venue}, {race}")
            return pd.DataFrame()
        
        file_path = files[0]
        delimiter = self._detect_delimiter(file_path)
        
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
            df['venue'] = venue
            df['race'] = race
            
            # Convert timestamp columns
            if 'TIME_UTC_SECONDS' in df.columns:
                df['datetime'] = pd.to_datetime(df['TIME_UTC_SECONDS'], unit='s', errors='coerce')
            
            logger.info(f"Loaded weather: {len(df)} rows from {file_path.name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading weather file {file_path}: {e}")
            return pd.DataFrame()
    
    def load_best_laps(self, venue: str, race: str) -> pd.DataFrame:
        """
        Load best 10 laps by driver file.
        
        Args:
            venue: Track venue name
            race: Race number ("Race 1" or "Race 2")
            
        Returns:
            DataFrame with best lap data
        """
        venue_path = self._get_venue_path(venue)
        pattern = f"**/*Best*Laps*{race}*.CSV"
        files = list(venue_path.glob(pattern))
        
        if not files:
            logger.warning(f"No best laps file found for {venue}, {race}")
            return pd.DataFrame()
        
        file_path = files[0]
        delimiter = self._detect_delimiter(file_path)
        
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
            df['venue'] = venue
            df['race'] = race
            
            # Parse all best lap time columns
            for i in range(1, 11):
                col = f'BESTLAP_{i}'
                if col in df.columns:
                    df[f'{col}_seconds'] = df[col].apply(self._parse_time_string)
            
            if 'AVERAGE' in df.columns:
                df['AVERAGE_seconds'] = df['AVERAGE'].apply(self._parse_time_string)
            
            logger.info(f"Loaded best laps: {len(df)} rows from {file_path.name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading best laps file {file_path}: {e}")
            return pd.DataFrame()
    
    def load_analysis_endurance(self, venue: str, race: str) -> pd.DataFrame:
        """
        Load analysis endurance with sections file.
        
        Args:
            venue: Track venue name
            race: Race number ("Race 1" or "Race 2")
            
        Returns:
            DataFrame with lap-by-lap analysis
        """
        venue_path = self._get_venue_path(venue)
        pattern = f"**/*Analysis*{race}*.CSV"
        files = list(venue_path.glob(pattern))
        
        if not files:
            logger.warning(f"No analysis file found for {venue}, {race}")
            return pd.DataFrame()
        
        file_path = files[0]
        delimiter = self._detect_delimiter(file_path)
        
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8')
            df['venue'] = venue
            df['race'] = race
            
            # Parse time columns
            time_cols = ['LAP_TIME', 'S1', 'S2', 'S3', 'ELAPSED']
            for col in time_cols:
                if col in df.columns:
                    df[f'{col}_seconds'] = df[col].apply(self._parse_time_string)
            
            logger.info(f"Loaded analysis: {len(df)} rows from {file_path.name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading analysis file {file_path}: {e}")
            return pd.DataFrame()
    
    def load_lap_timing(self, venue: str, race: str, timing_type: str = "time") -> pd.DataFrame:
        """
        Load lap timing data (time, start, or end).
        
        Args:
            venue: Track venue name
            race: Race number ("Race 1" or "Race 2")
            timing_type: "time", "start", or "end"
            
        Returns:
            DataFrame with lap timing data
        """
        venue_path = self._get_venue_path(venue)
        
        # Different venues have different naming patterns
        patterns = {
            'barber': f'R[12]_barber_lap_{timing_type}.csv',
            'COTA': f'COTA_lap_{timing_type}_R[12].csv',
            'indianapolis': f'R[12]_indianapolis_motor_speedway_lap_{timing_type}.csv',
            'virginia-international-raceway': f'vir_lap_{timing_type}_R[12].csv'
        }
        
        # Try to find the file (search recursively)
        file_path = None
        for pattern in patterns.values():
            files = list(venue_path.glob(f"**/{pattern}"))
            if files:
                # Filter by race number
                race_num = "1" if "Race 1" in race else "2"
                matching = [f for f in files if race_num in f.name]
                if matching:
                    file_path = matching[0]
                    break
        
        if not file_path:
            logger.warning(f"No lap {timing_type} file found for {venue}, {race}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            df['venue'] = venue
            df['race'] = race
            df['timing_type'] = timing_type
            
            # Parse timestamps
            timestamp_cols = ['timestamp', 'meta_time', 'value']
            for col in timestamp_cols:
                if col in df.columns:
                    df[f'{col}_parsed'] = pd.to_datetime(df[col], errors='coerce')
            
            logger.info(f"Loaded lap {timing_type}: {len(df)} rows from {file_path.name}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading lap {timing_type} file {file_path}: {e}")
            return pd.DataFrame()
    
    def load_telemetry(self, venue: str, race: str, sample_size: Optional[int] = 200000) -> pd.DataFrame:
        """
        Load telemetry data (can be very large).
        
        Args:
            venue: Track venue name
            race: Race number ("Race 1" or "Race 2")
            sample_size: If provided, only load first N rows (defaults to 200k for performance;
                         pass None to stream entire file)
            
        Returns:
            DataFrame with telemetry data
        """
        venue_path = self._get_venue_path(venue)
        
        # Different venues have different naming patterns
        patterns = {
            'barber': 'R[12]_barber_telemetry_data.csv',
            'COTA': 'R[12]_cota_telemetry_data.csv',
            'indianapolis': 'R[12]_indianapolis_motor_speedway_telemetry.csv',
            'virginia-international-raceway': 'R[12]_vir_telemetry_data.csv'
        }
        
        file_path = None
        for pattern in patterns.values():
            files = list(venue_path.glob(f"**/{pattern}"))
            if files:
                race_num = "1" if "Race 1" in race else "2"
                matching = [f for f in files if race_num in f.name]
                if matching:
                    file_path = matching[0]
                    break
        
        if not file_path:
            logger.warning(f"No telemetry file found for {venue}, {race}")
            return pd.DataFrame()
        
        try:
            if sample_size:
                df = pd.read_csv(file_path, nrows=sample_size, encoding='utf-8')
                logger.info(f"Loaded telemetry sample: {len(df)} rows from {file_path.name}")
            else:
                # Use chunking for large files
                chunk_list = []
                chunk_size = 100000
                for chunk in pd.read_csv(file_path, chunksize=chunk_size, encoding='utf-8'):
                    chunk_list.append(chunk)
                df = pd.concat(chunk_list, ignore_index=True)
                logger.info(f"Loaded telemetry: {len(df)} rows from {file_path.name}")
            
            df['venue'] = venue
            df['race'] = race
            
            # Parse timestamps
            timestamp_cols = ['timestamp', 'meta_time', 'expire_at']
            for col in timestamp_cols:
                if col in df.columns:
                    df[f'{col}_parsed'] = pd.to_datetime(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading telemetry file {file_path}: {e}")
            return pd.DataFrame()
    
    def load_championship_data(self) -> pd.DataFrame:
        """
        Load GR Drivers Championship file.
        
        Returns:
            DataFrame with championship standings
        """
        indy_path = self.base_path / "indianapolis"
        file_path = indy_path / "GR Drivers Championship-1.csv"
        
        if not file_path.exists():
            logger.warning(f"Championship file not found: {file_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"Loaded championship data: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading championship file {file_path}: {e}")
            return pd.DataFrame()
    
    def _get_venue_path(self, venue: str) -> Path:
        """
        Get the path to a venue's data directory.
        
        Args:
            venue: Venue name
            
        Returns:
            Path to venue directory
        """
        venue_mapping = {
            'barber': 'barber',
            'COTA': 'COTA',
            'indianapolis': 'indianapolis',
            'virginia-international-raceway': 'virginia-international-raceway/VIR',
            'VIR': 'virginia-international-raceway/VIR'
        }
        
        venue_dir = venue_mapping.get(venue, venue)
        
        # Check if it's in a Race subdirectory
        race_dirs = ['Race 1', 'Race 2']
        base_path = self.base_path / venue_dir
        
        # Return base path - individual methods will handle Race subdirectories
        return base_path
    
    def load_all_venues(self, file_type: str, race: str = "Race 1") -> Dict[str, pd.DataFrame]:
        """
        Load a specific file type for all venues.
        
        Args:
            file_type: Type of file to load
            race: Race number
            
        Returns:
            Dictionary mapping venue names to DataFrames
        """
        results = {}
        
        for venue in self.venues:
            try:
                if file_type == "results":
                    df = self.load_results_file(venue, race)
                elif file_type == "results_by_class":
                    df = self.load_results_by_class(venue, race)
                elif file_type == "weather":
                    df = self.load_weather_data(venue, race)
                elif file_type == "best_laps":
                    df = self.load_best_laps(venue, race)
                elif file_type == "analysis":
                    df = self.load_analysis_endurance(venue, race)
                elif file_type == "lap_timing":
                    df = self.load_lap_timing(venue, race)
                elif file_type == "telemetry":
                    df = self.load_telemetry(venue, race)  # Sample for speed
                else:
                    logger.warning(f"Unknown file type: {file_type}")
                    continue
                
                if not df.empty:
                    results[venue] = df
                    
            except Exception as e:
                logger.error(f"Error loading {file_type} for {venue}: {e}")
        
        return results


def validate_data_completeness(loader: DataLoader) -> pd.DataFrame:
    """
    Validate data completeness across all venues.
    
    Args:
        loader: DataLoader instance
        
    Returns:
        DataFrame with validation results
    """
    validation_results = []
    
    file_types = ["results", "results_by_class", "weather", "best_laps", "analysis"]
    races = ["Race 1", "Race 2"]
    
    for venue in loader.venues:
        for race in races:
            for file_type in file_types:
                try:
                    if file_type == "results":
                        df = loader.load_results_file(venue, race)
                    elif file_type == "results_by_class":
                        df = loader.load_results_by_class(venue, race)
                    elif file_type == "weather":
                        df = loader.load_weather_data(venue, race)
                    elif file_type == "best_laps":
                        df = loader.load_best_laps(venue, race)
                    elif file_type == "analysis":
                        df = loader.load_analysis_endurance(venue, race)
                    
                    validation_results.append({
                        'venue': venue,
                        'race': race,
                        'file_type': file_type,
                        'found': not df.empty,
                        'rows': len(df) if not df.empty else 0,
                        'columns': len(df.columns) if not df.empty else 0,
                        'missing_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100 if not df.empty else 100
                    })
                    
                except Exception as e:
                    validation_results.append({
                        'venue': venue,
                        'race': race,
                        'file_type': file_type,
                        'found': False,
                        'rows': 0,
                        'columns': 0,
                        'missing_pct': 100,
                        'error': str(e)
                    })
    
    # Ensure DataFrame has expected columns even if empty
    if not validation_results:
        return pd.DataFrame(columns=['venue', 'race', 'file_type', 'found', 'rows', 'columns', 'missing_pct'])
    
    return pd.DataFrame(validation_results)


def create_data_quality_report(loader: DataLoader, output_path: str = "data_quality_report.txt") -> str:
    """
    Create a comprehensive data quality report.
    
    Args:
        loader: DataLoader instance
        output_path: Path to save the report
        
    Returns:
        Report content as string
    """
    validation_df = validate_data_completeness(loader)
    
    report_lines = [
        "=" * 80,
        "GR CUP RACING DATA - QUALITY REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "SUMMARY",
        "-" * 80,
        f"Total Venues: {len(loader.venues)}",
        f"Total File Checks: {len(validation_df)}",
        f"Files Found: {validation_df['found'].sum()}",
        f"Files Missing: {(~validation_df['found']).sum()}",
        f"Overall Completeness: {validation_df['found'].mean() * 100:.1f}%",
        "",
        "DETAILED BREAKDOWN",
        "-" * 80,
    ]
    
    # Per venue breakdown
    for venue in loader.venues:
        venue_data = validation_df[validation_df['venue'] == venue]
        report_lines.append(f"\n{venue.upper()}:")
        report_lines.append(f"  Files Found: {venue_data['found'].sum()}/{len(venue_data)}")
        report_lines.append(f"  Total Rows: {venue_data['rows'].sum():,}")
        report_lines.append(f"  Average Missing Data: {venue_data['missing_pct'].mean():.2f}%")
    
    # Per file type breakdown
    report_lines.append("\n\nFILE TYPE BREAKDOWN:")
    report_lines.append("-" * 80)
    for file_type in validation_df['file_type'].unique():
        type_data = validation_df[validation_df['file_type'] == file_type]
        report_lines.append(f"\n{file_type.upper()}:")
        report_lines.append(f"  Files Found: {type_data['found'].sum()}/{len(type_data)}")
        report_lines.append(f"  Total Rows: {type_data['rows'].sum():,}")
    
    # Missing files
    missing = validation_df[~validation_df['found']]
    if not missing.empty:
        report_lines.append("\n\nMISSING FILES:")
        report_lines.append("-" * 80)
        for _, row in missing.iterrows():
            report_lines.append(f"  {row['venue']} - {row['race']} - {row['file_type']}")
    
    report_content = "\n".join(report_lines)
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Data quality report saved to {output_path}")
    return report_content


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Test loading a single file
    print("Testing data loader...")
    df = loader.load_results_file("barber", "Race 1", "provisional")
    print(f"Loaded {len(df)} rows from Barber Race 1 results")
    
    # Create data quality report
    print("\nGenerating data quality report...")
    report = create_data_quality_report(loader)
    print(report)

