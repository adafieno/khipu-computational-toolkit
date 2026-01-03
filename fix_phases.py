#!/usr/bin/env python3
"""Fix phase parameters from strings to integers"""
import re
from pathlib import Path

scripts_dir = Path("scripts")
# Match both single and double quotes
pattern = r'get_processed_file\(([^,)]+),\s*["\']phase(\d+)["\']\)'

def fix_phase(match):
    filename = match.group(1)
    phase_num = match.group(2)
    return f'get_processed_file({filename}, {phase_num})'

files_fixed = []
for script_file in scripts_dir.glob("*.py"):
    content = script_file.read_text(encoding='utf-8')
    new_content, count = re.subn(pattern, fix_phase, content)
    
    if count > 0:
        script_file.write_text(new_content, encoding='utf-8')
        files_fixed.append((script_file.name, count))
        print(f"Fixed {script_file.name}: {count} replacements")

print(f"\nTotal files fixed: {len(files_fixed)}")
