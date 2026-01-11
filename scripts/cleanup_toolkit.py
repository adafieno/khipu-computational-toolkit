# Cleanup Script - Remove temporary files and consolidate 3D viewer
# Run from project root: python scripts/cleanup_toolkit.py

import os
from pathlib import Path
import shutil

# Project root
ROOT = Path(__file__).parent.parent

# Files to delete (temporary/debug)
DELETE_FILES = [
    'scripts/check_colors.py',
    'scripts/check_hierarchy_cols.py',
    'scripts/check_khipu_colors.py',
    'scripts/check_knot_fields.py',
    'scripts/check_knot_type_docs.py',
    'scripts/check_schema.py',
    'scripts/explore_db_schema.py',
    'scripts/explore_knot_data.py',
    'scripts/debug_3d.py',
    'scripts/test_knot_rendering.py',
    'scripts/test_knot_spacing.py',
    'scripts/test_summation_single.py',
    'scripts/investigate_knot_values.py',
    'scripts/verify_ascher_notation.py',
    'scripts/temp_fix.txt',
    'scripts/analyze_summation_patterns.py',
    'scripts/test_alternative_summation.py',
    'scripts/test_hierarchical_summation.py',
    'scripts/test_summation_hypotheses.py',
    'scripts/generate_summation_report.py',
    'scripts/extract_pdf_text.py',
    'scripts/search_pdf_methodology.py',
]

# Files to deprecate (add notice at top)
DEPRECATE_FILES = [
    'scripts/visualize_3d_khipu.py',
    'scripts/interactive_3d_viewer.py',
]

DEPRECATION_NOTICE = '''"""
DEPRECATED: This viewer has been replaced by khipu_3d_viewer.py
This file is kept for historical reference only.
Please use: streamlit run scripts/khipu_3d_viewer.py
"""

'''

def main():
    print("=" * 60)
    print("Khipu Toolkit Cleanup")
    print("=" * 60)
    
    # Delete temporary files
    print("\n1. Removing temporary/debug files...")
    deleted_count = 0
    for file_path in DELETE_FILES:
        full_path = ROOT / file_path
        if full_path.exists():
            print(f"   Deleting: {file_path}")
            full_path.unlink()
            deleted_count += 1
        else:
            print(f"   (Not found, skipping: {file_path})")
    print(f"   Deleted {deleted_count} files")
    
    # Rename new 3D viewer to be THE 3D viewer
    print("\n2. Making plotly_3d_viewer_new.py the primary 3D viewer...")
    old_viewer = ROOT / 'scripts' / 'plotly_3d_viewer_new.py'
    new_viewer = ROOT / 'scripts' / 'khipu_3d_viewer.py'
    
    if old_viewer.exists():
        if new_viewer.exists():
            print("   Backing up existing khipu_3d_viewer.py...")
            backup = ROOT / 'scripts' / 'khipu_3d_viewer.py.backup'
            shutil.copy2(new_viewer, backup)
        
        print(f"   Renaming plotly_3d_viewer_new.py → khipu_3d_viewer.py")
        old_viewer.rename(new_viewer)
    else:
        print("   (plotly_3d_viewer_new.py not found)")
    
    # Add deprecation notices
    print("\n3. Adding deprecation notices to old viewers...")
    for file_path in DEPRECATE_FILES:
        full_path = ROOT / file_path
        if full_path.exists():
            print(f"   Deprecating: {file_path}")
            # Read existing content
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Only add notice if not already there
            if 'DEPRECATED' not in content[:500]:
                # Add notice at the top (after shebang if present)
                lines = content.split('\n')
                if lines[0].startswith('#!'):
                    # Keep shebang line
                    new_content = lines[0] + '\n' + DEPRECATION_NOTICE + '\n'.join(lines[1:])
                else:
                    new_content = DEPRECATION_NOTICE + content
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            else:
                print(f"      (Already deprecated)")
        else:
            print(f"   (Not found: {file_path})")
    
    # Clean up reports (keep only the two comprehensive plans)
    print("\n4. Organizing reports...")
    reports_to_keep = [
        'toolkit_integration_plan.md',
        'scholarly_rewrite_plan.md',
    ]
    reports_dir = ROOT / 'reports'
    if reports_dir.exists():
        report_files = [f for f in reports_dir.glob('*.md') if f.name not in reports_to_keep]
        report_files += [f for f in reports_dir.glob('*.txt')]
        
        # Create archive directory for old reports
        archive_dir = reports_dir / 'archive'
        archive_dir.mkdir(exist_ok=True)
        
        for report_file in report_files:
            if 'archive' not in str(report_file):
                print(f"   Archiving: {report_file.name}")
                shutil.move(str(report_file), str(archive_dir / report_file.name))
    
    # Summary
    print("\n" + "=" * 60)
    print("Cleanup Complete!")
    print("=" * 60)
    print("\nSummary:")
    print(f"  • Deleted {deleted_count} temporary files")
    print(f"  • Renamed plotly_3d_viewer_new.py → khipu_3d_viewer.py")
    print(f"  • Deprecated {len(DEPRECATE_FILES)} old viewers")
    print(f"  • Archived old reports to reports/archive/")
    print("\nNext Steps:")
    print("  1. Test 3D viewer: streamlit run scripts/khipu_3d_viewer.py")
    print("  2. Review CLEANUP_PLAN.md for details")
    print("  3. Begin Phase 1 implementation (value computation)")
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
