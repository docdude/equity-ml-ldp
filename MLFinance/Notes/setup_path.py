"""
Setup script to add the MLFinance root directory to Python path.
Import this at the beginning of any notebook in the Notes directory.

Usage:
    import setup_path
"""
import sys
from pathlib import Path

# Get the MLFinance root directory (two levels up from Notes)
project_root = Path(__file__).parent.parent.resolve()

# Add to Python path if not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Added {project_root} to Python path")
