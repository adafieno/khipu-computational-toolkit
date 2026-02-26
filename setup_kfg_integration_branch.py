#!/usr/bin/env python3
"""
KFG Integration Branch Setup

This script creates a new kfg-integration branch that will eventually replace main.
It reorganizes the repository to preserve OKR work as legacy documentation while
preparing for KFG as the primary data source.

Structure:
    OLD (main branch):
        data/processed/         <- OKR results
        reports/*.md            <- OKR reports
        visualizations/phase*/  <- OKR visualizations
    
    NEW (kfg-integration branch):
        data/processed/         <- KFG results (future)
        data/LEGACY_OKR/        <- OKR results (preserved)
        reports/*.md            <- KFG reports (future)
        reports/LEGACY_OKR/     <- OKR reports (preserved)
        visualizations/phase*/  <- KFG visualizations (future)
        visualizations/LEGACY_OKR/ <- OKR visualizations (preserved)

Usage:
    python setup_kfg_integration_branch.py
"""

import subprocess
import sys
import shutil
from pathlib import Path

def run_command(cmd, description, check=True):
    """Run a shell command and report results."""
    print(f"\n{'─'*70}")
    print(f"🔧 {description}")
    print(f"{'─'*70}")
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if check and result.returncode != 0:
        if result.stderr:
            print(f"❌ ERROR: {result.stderr}")
        sys.exit(1)
    return result.returncode == 0

def main():
    print("=" * 70)
    print("🚀 KFG Integration Branch Setup")
    print("=" * 70)
    print()
    print("This script will create a new kfg-integration branch that:")
    print("  1. Preserves OKR work in LEGACY_OKR/ folders")
    print("  2. Prepares structure for KFG as primary data source")
    print("  3. Will eventually replace main branch")
    print()
    
    # Step 1: Check git status
    print("\n📋 Step 1: Checking Git Status")
    print("─" * 70)
    
    result = subprocess.run(["git", "status", "--porcelain"], 
                          capture_output=True, text=True)
    if result.stdout.strip():
        print("⚠️  Warning: You have uncommitted changes:")
        print(result.stdout)
        response = input("\nCommit or stash them first. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting. Please commit or stash your changes first.")
            sys.exit(0)
    else:
        print("✅ Working directory is clean")
    
    # Step 2: Check current branch
    print("\n\n📋 Step 2: Checking Current Branch")
    print("─" * 70)
    
    result = subprocess.run(["git", "branch", "--show-current"],
                          capture_output=True, text=True)
    current_branch = result.stdout.strip()
    print(f"Current branch: {current_branch}")
    
    if current_branch != 'main':
        print(f"⚠️  You're not on 'main' branch")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting. Please switch to main first: git checkout main")
            sys.exit(0)
    
    # Step 3: Create kfg-integration branch
    print("\n\n📋 Step 3: Creating kfg-integration Branch")
    print("─" * 70)
    
    result = subprocess.run(["git", "branch", "--list", "kfg-integration"],
                          capture_output=True, text=True)
    if result.stdout.strip():
        print("⚠️  Branch 'kfg-integration' already exists")
        response = input("Delete and recreate it? (y/n): ")
        if response.lower() == 'y':
            run_command("git branch -D kfg-integration", "Deleting existing branch")
        else:
            print("Using existing branch")
            run_command("git checkout kfg-integration", "Switching to existing branch")
            return
    
    run_command("git checkout -b kfg-integration",
               "Creating and switching to kfg-integration branch")
    
    # Step 4: Reorganize directory structure
    print("\n\n📋 Step 4: Reorganizing for Legacy Preservation")
    print("─" * 70)
    print()
    print("This will:")
    print("  • Move data/processed/ → data/LEGACY_OKR/")
    print("  • Move reports/phase*.md → reports/LEGACY_OKR/")
    print("  • Move visualizations/ → visualizations/LEGACY_OKR/")
    print()
    response = input("Proceed with reorganization? (y/n): ")
    if response.lower() != 'y':
        print("Skipping reorganization. You'll need to do this manually.")
        return
    
    root = Path.cwd()
    
    # Move processed data
    print("\n📦 Moving processed data to LEGACY_OKR...")
    if (root / "data" / "processed").exists():
        legacy_data = root / "data" / "LEGACY_OKR"
        legacy_data.mkdir(exist_ok=True)
        shutil.move(str(root / "data" / "processed"), 
                   str(legacy_data / "processed"))
        print(f"✅ Moved: data/processed/ → data/LEGACY_OKR/processed/")
    
    # Move reports
    print("\n📄 Moving reports to LEGACY_OKR...")
    reports_dir = root / "reports"
    legacy_reports = reports_dir / "LEGACY_OKR"
    legacy_reports.mkdir(exist_ok=True)
    
    moved_reports = []
    for report in reports_dir.glob("phase*.md"):
        shutil.move(str(report), str(legacy_reports / report.name))
        moved_reports.append(report.name)
    
    if moved_reports:
        print(f"✅ Moved {len(moved_reports)} reports to reports/LEGACY_OKR/")
        for r in moved_reports[:5]:  # Show first 5
            print(f"   • {r}")
        if len(moved_reports) > 5:
            print(f"   ... and {len(moved_reports) - 5} more")
    
    # Move visualizations
    print("\n🎨 Moving visualizations to LEGACY_OKR...")
    viz_dir = root / "visualizations"
    legacy_viz = root / "visualizations_LEGACY_OKR"
    
    if viz_dir.exists():
        # Move everything except README.md
        legacy_viz.mkdir(exist_ok=True)
        moved_viz = []
        for item in viz_dir.iterdir():
            if item.name != "README.md" and item.is_dir():
                shutil.move(str(item), str(legacy_viz / item.name))
                moved_viz.append(item.name)
        
        if moved_viz:
            print(f"✅ Moved {len(moved_viz)} visualization folders")
            for v in moved_viz[:5]:
                print(f"   • {v}")
            if len(moved_viz) > 5:
                print(f"   ... and {len(moved_viz) - 5} more")
    
    # Step 5: Create new directory structure
    print("\n\n📋 Step 5: Creating New Directory Structure for KFG")
    print("─" * 70)
    
    # Create fresh directories
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (root / "visualizations").mkdir(exist_ok=True)
    
    print("✅ Created fresh directories for KFG data")
    
    # Step 6: Commit changes
    print("\n\n📋 Step 6: Committing Branch Changes")
    print("─" * 70)
    
    run_command("git add -A", "Staging all changes")
    run_command(
        'git commit -m "refactor: Reorganize for KFG integration\n\n' +
        '- Move OKR results to LEGACY_OKR/ folders\n' +
        '- Create fresh structure for KFG primary data\n' +
        '- Preserve OKR work as methodology validation\n' +
        'This branch will eventually replace main."',
        "Committing reorganization"
    )
    
    # Step 7: Summary
    print("\n\n" + "=" * 70)
    print("✅ KFG Integration Branch Setup Complete!")
    print("=" * 70)
    print()
    print("📂 New Structure:")
    print("   data/")
    print("   ├── LEGACY_OKR/processed/    ← OKR data (preserved)")
    print("   ├── processed/               ← KFG data (empty, ready)")
    print("   └── kfg/khipu_database.db    ← KFG database")
    print()
    print("   reports/")
    print("   ├── LEGACY_OKR/*.md          ← OKR reports (preserved)")
    print("   └── (ready for KFG reports)")
    print()
    print("   visualizations_LEGACY_OKR/   ← OKR visualizations (preserved)")
    print("   visualizations/              ← KFG visualizations (empty, ready)")
    print()
    print("📋 Next Steps:")
    print()
    print("1. Run KFG pipeline scripts:")
    print("   python scripts/extract_cord_hierarchy.py --kfg")
    print("   python scripts/extract_knot_data.py --kfg")
    print("   python scripts/extract_color_data.py --kfg")
    print("   (etc... through all phases)")
    print()
    print("2. Generate new KFG reports and visualizations")
    print()
    print("3. Create comparison document showing OKR vs KFG")
    print()
    print("4. When complete and validated:")
    print("   git checkout main")
    print("   git merge kfg-integration")
    print("   (This replaces main with KFG version)")
    print()
    print("5. Optional: Keep old main as archive:")
    print("   git branch -m main okr-archive")
    print("   git branch -m kfg-integration main")
    print()
    print("=" * 70)
    print()
    print("📖 See docs/KFG_MIGRATION_STRATEGY.md for full details")
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
