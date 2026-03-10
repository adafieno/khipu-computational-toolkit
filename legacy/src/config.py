"""
Centralized Configuration Management for Khipu Toolkit

This module provides a single source of truth for all file paths, database locations,
and configuration settings used throughout the toolkit.

Environment Variables:
    KHIPU_DB_PATH: Path to the Open Khipu Repository database file
                   Default: ../open-khipu-repository/data/khipu.db
    
Usage:
    from src.config import Config
    
    config = Config()
    db_path = config.get_database_path()
    hierarchy_path = config.get_processed_file('cord_hierarchy.csv')
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Central configuration for all toolkit paths and settings."""
    
    def __init__(self, root_dir: Optional[Path] = None):
        """
        Initialize configuration.
        
        Args:
            root_dir: Optional root directory. If None, auto-detects from this file's location.
        """
        if root_dir is None:
            # Auto-detect: config.py is in src/, so parent is root
            self.root_dir = Path(__file__).parent.parent.resolve()
        else:
            self.root_dir = Path(root_dir).resolve()
        
        # Core directories
        self.src_dir = self.root_dir / "src"
        self.scripts_dir = self.root_dir / "scripts"
        self.data_dir = self.root_dir / "data"
        self.docs_dir = self.root_dir / "docs"
        self.reports_dir = self.root_dir / "reports"
        self.outputs_dir = self.root_dir / "outputs"
        self.models_dir = self.root_dir / "models"
        self.notebooks_dir = self.root_dir / "notebooks"
        self.visualizations_dir = self.root_dir / "visualizations"
        
        # Data subdirectories
        self.processed_dir = self.data_dir / "processed"
        self.graphs_dir = self.data_dir / "graphs"
        
        # Phase-specific output directories
        self.phase_dirs = {
            0: self.processed_dir / "phase0",
            1: self.processed_dir / "phase1",
            2: self.processed_dir / "phase2",
            3: self.processed_dir / "phase3",
            4: self.processed_dir / "phase4",
            5: self.processed_dir / "phase5",
            7: self.processed_dir / "phase7",
            8: self.processed_dir / "phase8",
            9: self.processed_dir / "phase9",
        }
    
    def get_database_path(self) -> Path:
        """
        Get path to the Open Khipu Repository database.
        
        Returns database path in this order of precedence:
        1. KHIPU_DB_PATH environment variable
        2. Default: ../open-khipu-repository/data/khipu.db (relative to toolkit root)
        
        Returns:
            Path object pointing to khipu.db
        """
        env_path = os.environ.get('KHIPU_DB_PATH')
        if env_path:
            return Path(env_path).resolve()
        
        # Default: sibling directory structure
        default_path = self.root_dir.parent / "open-khipu-repository" / "data" / "khipu.db"
        return default_path.resolve()
    
    def get_processed_file(self, filename: str, phase: Optional[int] = None) -> Path:
        """
        Get path to a processed data file.
        
        Args:
            filename: Name of the file (e.g., 'cord_hierarchy.csv')
            phase: Optional phase number. If provided, looks in data/processed/phaseN/
                   If None, looks in data/processed/ (root level)
        
        Returns:
            Full path to the processed file
        """
        if phase is not None and phase in self.phase_dirs:
            return self.phase_dirs[phase] / filename
        else:
            return self.processed_dir / filename
    
    def get_output_file(self, filename: str, subdir: Optional[str] = None) -> Path:
        """
        Get path for an output file.
        
        Args:
            filename: Name of the output file
            subdir: Optional subdirectory within outputs/ (e.g., 'visualizations')
        
        Returns:
            Full path to the output file
        """
        if subdir:
            return self.outputs_dir / subdir / filename
        else:
            return self.outputs_dir / filename
    
    def get_visualization_dir(self, phase: str) -> Path:
        """
        Get visualization directory for a specific phase.
        
        Args:
            phase: Phase identifier (e.g., 'phase1_baseline', 'phase8_comparative')
        
        Returns:
            Path to the phase-specific visualization directory
        """
        return self.visualizations_dir / phase
    
    def ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.processed_dir,
            self.graphs_dir,
            self.outputs_dir,
            self.outputs_dir / "visualizations",
            self.models_dir,
            self.visualizations_dir,
        ]
        
        # Add all phase directories
        directories.extend(self.phase_dirs.values())
        
        # Add visualization phase directories
        viz_phases = [
            "phase1_baseline",
            "phase2_extraction",
            "phase3_summation",
            "phase4_patterns",
            "phase5_multimodel",
            "phase7_ml",
            "phase8_comparative",
            "phase9_stability",
        ]
        directories.extend([self.visualizations_dir / phase for phase in viz_phases])
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate_setup(self) -> dict:
        """
        Validate that the toolkit is properly configured.
        
        Returns:
            Dictionary with validation results and any warnings/errors
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'info': {}
        }
        
        # Check database path
        db_path = self.get_database_path()
        results['info']['database_path'] = str(db_path)
        
        if not db_path.exists():
            results['valid'] = False
            results['errors'].append(
                f"Database not found at: {db_path}\n"
                f"Please clone open-khipu-repository as a sibling directory, "
                f"or set KHIPU_DB_PATH environment variable."
            )
        
        # Check root directory structure
        if not self.src_dir.exists():
            results['errors'].append(f"src/ directory not found at: {self.src_dir}")
            results['valid'] = False
        
        if not self.scripts_dir.exists():
            results['warnings'].append(f"scripts/ directory not found at: {self.scripts_dir}")
        
        # Check if processed directory exists (will be created by scripts)
        if not self.processed_dir.exists():
            results['warnings'].append(
                f"data/processed/ directory not found. "
                f"It will be created when you run extraction scripts."
            )
        
        results['info']['root_directory'] = str(self.root_dir)
        results['info']['processed_directory'] = str(self.processed_dir)
        
        return results


# Global singleton instance
_config = None

def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def validate_and_report():
    """Validate configuration and print a report."""
    config = get_config()
    results = config.validate_setup()
    
    print("=" * 70)
    print("Khipu Toolkit Configuration Validation")
    print("=" * 70)
    print()
    
    print("📁 Paths:")
    for key, value in results['info'].items():
        print(f"  {key}: {value}")
    print()
    
    if results['errors']:
        print("❌ Errors:")
        for error in results['errors']:
            print(f"  • {error}")
        print()
    
    if results['warnings']:
        print("⚠️  Warnings:")
        for warning in results['warnings']:
            print(f"  • {warning}")
        print()
    
    if results['valid'] and not results['warnings']:
        print("✅ Configuration is valid!")
    elif results['valid']:
        print("✅ Configuration is valid (with warnings)")
    else:
        print("❌ Configuration has errors that must be fixed")
    
    print("=" * 70)
    
    return results['valid']


if __name__ == "__main__":
    # Run validation when executed directly
    import sys
    sys.exit(0 if validate_and_report() else 1)
