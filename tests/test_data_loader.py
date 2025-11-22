"""
Unit tests for DataLoader
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data_processing.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class"""
    
    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance for testing"""
        return DataLoader()
    
    def test_loader_initialization(self, loader):
        """Test that loader initializes correctly"""
        assert loader is not None
        assert loader.base_path is not None
        assert len(loader.venues) > 0
    
    def test_load_results_file(self, loader):
        """Test loading results file"""
        df = loader.load_results_file("barber", "Race 1", "provisional")
        assert isinstance(df, pd.DataFrame)
        # If file exists, should have data
        if not df.empty:
            assert 'venue' in df.columns
            assert 'race' in df.columns
    
    def test_load_weather_data(self, loader):
        """Test loading weather data"""
        df = loader.load_weather_data("barber", "Race 1")
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert 'venue' in df.columns
            assert 'race' in df.columns
    
    def test_parse_time_string(self, loader):
        """Test time string parsing"""
        # Test MM:SS.mmm format
        result = loader._parse_time_string("1:37.428")
        assert result is not None
        assert result > 0
        
        # Test invalid input
        result = loader._parse_time_string("")
        assert result is None
        
        result = loader._parse_time_string("-")
        assert result is None
    
    def test_detect_delimiter(self, loader):
        """Test delimiter detection"""
        # This would require actual files, so we'll test the logic
        # In practice, semicolon files are .CSV, comma files are .csv
        pass  # Placeholder for actual file-based tests
    
    def test_venue_path_mapping(self, loader):
        """Test venue path mapping"""
        path = loader._get_venue_path("barber")
        assert path is not None
        assert isinstance(path, Path)








