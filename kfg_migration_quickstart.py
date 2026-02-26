#!/usr/bin/env python3
"""
Quick Start: KFG Migration

This script helps you get started with the KFG data migration by:
1. Validating both OKR and KFG configurations
2. Creating the kfg-integration branch
3. Setting up KFG directory structure
4. Providing next steps

Usage:
    python kfg_migration_quickstart.py
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and report results."""
    print(f"\n{'='*70}")
    print(f"🔧 {description}")
    print(f"{'='*70}")
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f"⚠️  Warning: {result.stderr}")
    return result.returncode == 0

def main():
    print("=" * 70)
    print("🚀 KFG Migration Quick Start")
    print("=" * 70)
    print()
    print("This script will set up your environment for KFG data migration.")
    print()
    
    # Step 1: Validate configurations
    print("\n📋 Step 1: Validating Configurations")
    print("-" * 70)
    
    try:
        from src.config import get_config
        from src.config_kfg import get_kfg_config
        
        print("\n✅ OKR Configuration:")
        okr_config = get_config()
        okr_db = okr_config.get_database_path()
        print(f"   Database: {okr_db}")
        print(f"   Exists: {'✅ Yes' if okr_db.exists() else '❌ No'}")
        
        print("\n✅ KFG Configuration:")
        kfg_config = get_kfg_config()
        kfg_db = kfg_config.get_database_path()
        print(f"   Database: {kfg_db}")
        print(f"   Exists: {'✅ Yes' if kfg_db.exists() else '❌ No'}")
        
        if not kfg_db.exists():
            print("\n❌ ERROR: KFG database not found!")
            print(f"   Please ensure {kfg_db} exists before continuing.")
            sys.exit(1)
            
    except ImportError as e:
        print(f"\n❌ ERROR: Could not import configuration modules")
        print(f"   {e}")
        sys.exit(1)
    
    # Step 2: Check git status
    print("\n\n📋 Step 2: Checking Git Status")
    print("-" * 70)
    
    result = subprocess.run(["git", "status", "--porcelain"], 
                          capture_output=True, text=True)
    if result.stdout.strip():
        print("⚠️  Warning: You have uncommitted changes:")
        print(result.stdout)
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            sys.exit(0)
    else:
        print("✅ Working directory is clean")
    
    # Step 3: Check if kfg-integration branch exists
    print("\n\n📋 Step 3: Checking for kfg-integration Branch")
    print("-" * 70)
    
    result = subprocess.run(["git", "branch", "--list", "kfg-integration"],
                          capture_output=True, text=True)
    branch_exists = bool(result.stdout.strip())
    
    if branch_exists:
        print("📌 Branch 'kfg-integration' already exists")
        response = input("Switch to it? (y/n): ")
        if response.lower() == 'y':
            run_command("git checkout kfg-integration", 
                       "Switching to kfg-integration branch")
        else:
            print("Staying on current branch")
    else:
        print("📌 Branch 'kfg-integration' does not exist")
        response = input("Create and switch to kfg-integration branch? (y/n): ")
        if response.lower() == 'y':
            success = run_command("git checkout -b kfg-integration",
                                "Creating kfg-integration branch")
            if success:
                print("\n✅ Successfully created and switched to kfg-integration branch")
        else:
            print("Skipping branch creation")
    
    # Step 4: Create KFG directory structure
    print("\n\n📋 Step 4: Setting Up KFG Directory Structure")
    print("-" * 70)
    
    kfg_config.ensure_directories()
    print("✅ Created KFG-specific directories:")
    print(f"   {kfg_config.kfg_processed_dir}/")
    for phase_name, phase_dir in kfg_config.kfg_phase_dirs.items():
        print(f"   {phase_dir.relative_to(kfg_config.root_dir)}/")
    
    # Step 5: Summary and next steps
    print("\n\n" + "=" * 70)
    print("✅ KFG Migration Setup Complete!")
    print("=" * 70)
    print()
    print("📋 Next Steps:")
    print()
    print("1. Update extraction scripts to support --kfg flag:")
    print("   python scripts/extract_cord_hierarchy.py --kfg")
    print()
    print("2. Run complete KFG pipeline:")
    print("   See: docs/KFG_MIGRATION_STRATEGY.md")
    print()
    print("3. Compare OKR vs KFG results:")
    print("   - OKR data:  data/processed/")
    print("   - KFG data:  data/processed_kfg/")
    print()
    print("4. Validate and document differences")
    print()
    print("5. Once validated, merge to main:")
    print("   git checkout main")
    print("   git merge kfg-integration")
    print()
    print("=" * 70)
    print()
    print("📖 Full migration guide: docs/KFG_MIGRATION_STRATEGY.md")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
