#!/usr/bin/env python3
"""
Integration tests for the ILP entropy package.
Tests the full CLI workflow with parameter sweeps.
"""

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

def test_entropy_sweep_integration():
    """Test the entropy sweep with a small example."""
    
    # Create a temporary directory for test output
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a test word file in temp directory
        test_words = ["that", "with", "have", "this", "they"]  # all 4+ letters
        temp_file = temp_path / "test_words.txt"
        
        with open(temp_file, 'w') as f:
            for word in test_words:
                f.write(f"{word}\n")
        
        print(f"Created test file with words: {test_words}")
        
        # Run a small parameter sweep from project root, output to temp directory
        cmd = [
            "poetry", "run", "python", "-m", "ilp_entropy",
            "--verbose",
            "sweep", str(temp_file),
            "--drop-left-range", "0.1", "0.2", "0.05",   # just 0.1, 0.15, 0.2
            "--drop-right-range", "0.1", "0.2", "0.05",  # just 0.1, 0.15, 0.2
            "--output-dir", str(temp_path)  # Output to temp directory
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)  # Run from current directory (project root)
        
        # Verify some output was created
        output_files = list(temp_path.glob("*.csv"))
        if output_files:
            print(f"✓ Created {len(output_files)} output files in temp directory")
        else:
            print("⚠ No output files found")
        
        # Temp directory and all contents automatically cleaned up
        return result.returncode

def main():
    """Run integration tests."""
    return test_entropy_sweep_integration()

if __name__ == "__main__":
    sys.exit(main()) 