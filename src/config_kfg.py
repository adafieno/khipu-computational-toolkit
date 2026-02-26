"""
KFG Data Source Configuration Extension

This module extends the base Config class to support the Khipu Field Guide (KFG)
database as an alternative data source while preserving backward compatibility
with Open Khipu Repository (OKR) data.

Usage:
    # Use KFG data source
    from src.config_kfg import get_kfg_config
    config = get_kfg_config()
    
    # Or via environment variable
    export KHIPU_DATA_SOURCE=kfg
    from src.config import get_config
    config = get_config()
"""

import os
from pathlib import Path
from typing import Optional
from .config import Config


class KFGConfig(Config):
    """Extended configuration for KFG data source."""
    
    def __init__(self, root_dir: Optional[Path] = None):
        """Initialize KFG-specific configuration."""
        super().__init__(root_dir)
        self.data_source = 'kfg'
        
        # KFG-specific directories (separate from OKR processed data)
        self.kfg_processed_dir = self.data_dir / "processed_kfg"
        self.kfg_phase_dirs = {
            phase: self.kfg_processed_dir / f"phase{phase}"
            for phase in [0, 1, 2, 3, 4, 5, 7, 8, 9]
        }
    
    def get_database_path(self) -> Path:
        """
        Get path to KFG database.
        
        Returns:
            Path to data/kfg/khipu_database.db
        """
        env_path = os.environ.get('KHIPU_DB_PATH')
        if env_path:
            return Path(env_path).resolve()
        
        # KFG database is within toolkit repository
        kfg_db_path = self.data_dir / "kfg" / "khipu_database.db"
        return kfg_db_path.resolve()
    
    def get_processed_file(self, filename: str, phase: Optional[int] = None) -> Path:
        """
        Get path to KFG processed data file.
        
        Files are stored in data/processed_kfg/ to avoid overwriting
        OKR-derived results during migration.
        
        Args:
            filename: Name of the file (e.g., 'cord_hierarchy.csv')
            phase: Optional phase number
        
        Returns:
            Full path to the KFG processed file
        """
        if phase is not None and phase in self.kfg_phase_dirs:
            return self.kfg_phase_dirs[phase] / filename
        else:
            return self.kfg_processed_dir / filename
    
    def ensure_directories(self):
        """Create all necessary directories including KFG-specific ones."""
        # Create base directories
        super().ensure_directories()
        
        # Create KFG-specific directories
        directories = [
            self.kfg_processed_dir,
            *self.kfg_phase_dirs.values()
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate_setup(self) -> dict:
        """
        Validate KFG-specific configuration.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'info': {}
        }
        
        # Check KFG database path
        db_path = self.get_database_path()
        results['info']['data_source'] = 'KFG (Khipu Field Guide)'
        results['info']['database_path'] = str(db_path)
        
        if not db_path.exists():
            results['valid'] = False
            results['errors'].append(
                f"KFG database not found at: {db_path}\n"
                f"Expected location: data/kfg/khipu_database.db\n"
                f"Please ensure the KFG database is present."
            )
        else:
            # Check database size (should be substantial)
            size_mb = db_path.stat().st_size / (1024 * 1024)
            results['info']['database_size'] = f"{size_mb:.1f} MB"
            if size_mb < 1:
                results['warnings'].append(
                    f"KFG database seems small ({size_mb:.1f} MB). "
                    f"Verify it contains complete data."
                )
        
        # Check root directory structure
        if not self.src_dir.exists():
            results['errors'].append(f"src/ directory not found at: {self.src_dir}")
            results['valid'] = False
        
        # Note that processed_kfg will be created by scripts
        if not self.kfg_processed_dir.exists():
            results['info']['kfg_processed'] = 'Will be created on first run'
        else:
            results['info']['kfg_processed'] = str(self.kfg_processed_dir)
        
        results['info']['root_directory'] = str(self.root_dir)
        
        return results


# KFG-specific singleton
_kfg_config = None


def get_kfg_config() -> KFGConfig:
    """
    Get the KFG configuration instance.
    
    Returns:
        KFGConfig instance configured for KFG data source
    """
    global _kfg_config
    if _kfg_config is None:
        _kfg_config = KFGConfig()
    return _kfg_config


def validate_kfg_setup():
    """Validate KFG configuration and print a report."""
    config = get_kfg_config()
    results = config.validate_setup()
    
    print("=" * 70)
    print("KFG (Khipu Field Guide) Configuration Validation")
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
        print("✅ KFG Configuration is valid!")
    elif results['valid']:
        print("✅ KFG Configuration is valid (with warnings)")
    else:
        print("❌ KFG Configuration has errors that must be fixed")
    
    print("=" * 70)
    
    return results['valid']


if __name__ == "__main__":
    # Run KFG validation when executed directly
    import sys
    sys.exit(0 if validate_kfg_setup() else 1)
