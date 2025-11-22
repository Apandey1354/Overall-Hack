"""
Script to run data validation and generate quality report
Run this to validate all data files and create a comprehensive report
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.data_loader import DataLoader, create_data_quality_report

def main():
    """Run data validation"""
    print("=" * 80)
    print("GR CUP RACING DATA - VALIDATION SCRIPT")
    print("=" * 80)
    print()
    
    # Initialize loader
    print("Initializing data loader...")
    loader = DataLoader()
    print(f"Found {len(loader.venues)} venues: {', '.join(loader.venues)}")
    print()
    
    # Generate quality report
    print("Generating data quality report...")
    print("This may take a few minutes as we check all files...")
    print()
    
    report = create_data_quality_report(loader, output_path="data_quality_report.txt")
    
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print()
    print("Report saved to: data_quality_report.txt")
    print()
    print("Quick Summary:")
    print("-" * 80)
    print(report.split("SUMMARY")[1].split("DETAILED")[0] if "SUMMARY" in report else "See full report for details")

if __name__ == "__main__":
    main()

