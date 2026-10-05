"""Script to check for cheating in the student's submission."""

import sys
from pathlib import Path

student_dir = Path("student_code")
base_dir = Path("original_main")  # The competition root directory

# 1. Check files that must be EXACT matches
exact_files = ["scripts/evaluate.py", "scripts/sim.py", ".github/workflows/competition.yml"]

for file_path in exact_files:
    student_file = student_dir / file_path
    base_file = base_dir / file_path

    if not student_file.exists():
        # Special formatting for GitHub Actions logs to show an error message
        print(f"::error title=Anti-Cheat Failure::🚨 CHEAT DETECTED: {file_path} was deleted!")
        sys.exit(1)  # Ends the action

    if student_file.read_text() != base_file.read_text():
        # Special formatting for GitHub Actions logs to show an error message
        print(
            f"::error title=Anti-Cheat Failure::🚨 CHEAT DETECTED: {file_path} has been modified!"
        )
        sys.exit(1)  # Ends the action

# 2. Check the TOML file (Only lines 7 and 43 allowed to change)
toml_path = "config/level2.toml"
student_toml_file = student_dir / toml_path
base_toml_file = base_dir / toml_path

if not student_toml_file.exists():
    # Special formatting for GitHub Actions logs to show an error message
    print(f"::error title=Anti-Cheat Failure::🚨 CHEAT DETECTED: {toml_path} was deleted!")
    sys.exit(1)  # Ends the action

student_lines = student_toml_file.read_text().splitlines()
base_lines = base_toml_file.read_text().splitlines()

# If they added or removed lines, the line numbers would shift, which is a violation
if len(student_lines) != len(base_lines):
    # Special formatting for GitHub Actions logs to show an error message
    print(
        "::error title=Anti-Cheat Failure::🚨 CHEAT DETECTED: "
        + f"{toml_path} has a different number of lines!"
    )
    sys.exit(1)  # Ends the action

# Allowed lines are 1-indexed (7 and 48), so their list indices are 6 and 47
allowed_indices = {6, 47}

for i, (s_line, b_line) in enumerate(zip(student_lines, base_lines)):
    if i not in allowed_indices and s_line != b_line:
        error_msg = (
            f"🚨 CHEAT DETECTED: {toml_path} was modified on line {i + 1}! "
            f"Expected: '{b_line.strip()}' | Found: '{s_line.strip()}'"
        )
        # Special formatting for GitHub Actions logs to show an error message
        print(f"::error title=Anti-Cheat Failure::{error_msg}")
        sys.exit(1)  # Ends the action

print("✅ Anti-cheat checks passed. No unauthorized modifications detected.")
