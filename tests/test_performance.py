#!/usr/bin/env python3
"""
Performance and large-scale integration tests for the ILP entropy package.

These tests exercise the package with realistic workloads using the corpus data.
They may take longer to run and are intended to validate performance and
scalability with larger datasets.
"""

import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest

from ilp_entropy import get_top_words
from ilp_entropy.runner import batch_ilp_entropy, parameter_sweep


def progress_reporter(completed: int, total: int) -> None:
    """Progress callback that shows detailed progress."""
    if total > 0:
        percent = 100 * completed / total
        bar_length = 40
        filled_length = int(bar_length * completed // total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\r  Progress: [{bar}] {completed}/{total} ({percent:.1f}%) words", end="", flush=True)
        if completed == total:
            print()  # New line when complete


class TestPerformance:
    """Performance tests with larger datasets."""
    
    def test_top_1k_words_entropy(self):
        """Test entropy calculation for top 1000 most frequent words."""
        print("\n🚀 Testing entropy calculation for top 1000 words...")
        
        # Load top 1000 words with error handling
        print("  📚 Loading corpus data...")
        start_time = time.time()
        try:
            words = get_top_words(1000, word_lengths=range(4, 8))  # 4-7 letter words only
            load_time = time.time() - start_time
            
            print(f"  ✓ Loaded {len(words)} words in {load_time:.2f}s")
            assert len(words) > 0, "Should load at least some words"
            assert all(4 <= len(word) <= 7 for word in words), "Words should be 4-7 letters"
            
            # Show sample of loaded words
            print(f"  📝 Sample words: {words[:10]}")
            
        except Exception as e:
            print(f"  ❌ Failed to load corpus data: {e}")
            raise
        
        # Test batch entropy calculation with progress
        print("  🧮 Calculating entropy for all words...")
        start_time = time.time()
        try:
            df = batch_ilp_entropy(
                words,
                drop_left=0.2,
                drop_right=0.3,
                n_workers=6,  # Reasonable number for most systems
                progress_callback=progress_reporter
            )
            calc_time = time.time() - start_time
            
            print(f"  ✓ Calculated entropy for {len(words)} words in {calc_time:.2f}s")
            print(f"  ✓ Processing rate: {len(words) / calc_time:.1f} words/sec")
            
        except Exception as e:
            print(f"  ❌ Failed during entropy calculation: {e}")
            print(f"  🔍 Error details: {type(e).__name__}")
            raise
        
        # Validate output format
        print("  🔍 Validating results...")
        expected_columns = {"word", "position", "entropy", "drop_left", "drop_right"}
        assert set(df.columns) == expected_columns, f"Unexpected columns: {df.columns}"
        
        # Validate data content
        assert len(df) > 0, "Should produce results"
        assert all(df["drop_left"] == 0.2), "Drop left should be consistent"
        assert all(df["drop_right"] == 0.3), "Drop right should be consistent"
        assert all(df["entropy"] > 0), "Entropy should be positive"
        assert all(df["position"] >= 1), "Positions should be 1-indexed"
        
        # Check that we have multiple positions per word
        positions_per_word = df.groupby("word")["position"].count()
        assert all(positions_per_word >= 4), "Should have multiple positions per word"
        
        # Show some statistics
        mean_entropy = df["entropy"].mean()
        std_entropy = df["entropy"].std()
        print(f"  📊 Entropy statistics: mean={mean_entropy:.3f}, std={std_entropy:.3f}")
        print(f"  ✓ Validated {len(df)} entropy calculations")
        print("  🎉 Top 1000 words test passed!")
    
    def test_parameter_sweep_medium_scale(self):
        """Test parameter sweep with 1000 words and 5x5 parameter grid."""
        print("\n🔬 Testing parameter sweep with 1000 words and 5×5 grid...")
        
        # Use top 1000 words for comprehensive parameter sweep
        print("  📚 Loading 1000 words for parameter sweep...")
        try:
            words = get_top_words(1000, word_lengths=range(4, 8))  # 4-7 letter words
            print(f"  ✓ Loaded {len(words)} words for parameter sweep")
            print(f"  📝 Sample words: {words[:5]}")
        except Exception as e:
            print(f"  ❌ Failed to load words: {e}")
            raise
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Define 5x5 parameter grid
            drop_left_values = [0.1, 0.2, 0.3, 0.4, 0.5]
            drop_right_values = [0.1, 0.2, 0.3, 0.4, 0.5]
            total_combinations = len(drop_left_values) * len(drop_right_values)
            total_computations = len(words) * total_combinations
            
            print(f"  🔄 Running parameter sweep: {len(drop_left_values)}×{len(drop_right_values)} = {total_combinations} combinations")
            print(f"  📊 Total computations: {len(words)} words × {total_combinations} combinations = {total_computations:,}")
            
            # Enhanced progress reporter for parameter sweep
            def sweep_progress_reporter(completed: int, total: int) -> None:
                """Progress callback that shows detailed sweep progress."""
                if total > 0:
                    percent = 100 * completed / total
                    bar_length = 50
                    filled_length = int(bar_length * completed // total)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    
                    # Calculate which combination we're in
                    combo_num = (completed - 1) // len(words) + 1 if completed > 0 else 0
                    words_in_combo = completed % len(words) if completed % len(words) != 0 else len(words)
                    
                    print(f"\r  Sweep: [{bar}] {completed:,}/{total:,} ({percent:.1f}%) "
                          f"| Combo {combo_num}/{total_combinations} "
                          f"| Words {words_in_combo}/{len(words)}", end="", flush=True)
                    if completed == total:
                        print()  # New line when complete
            
            # Run parameter sweep with 5x5 grid and progress tracking
            start_time = time.time()
            try:
                df = parameter_sweep(
                    words,
                    drop_left_values=drop_left_values,
                    drop_right_values=drop_right_values,
                    output_dir=temp_path,
                    n_workers=6,  # Use reasonable number of workers
                    progress_callback=sweep_progress_reporter
                )
                sweep_time = time.time() - start_time
                
                computation_rate = total_computations / sweep_time
                
                print(f"  ✓ Parameter sweep completed in {sweep_time:.1f}s")
                print(f"  ✓ Processing rate: {computation_rate:.1f} word-param-combinations/sec")
                print(f"  ✓ Average per combination: {sweep_time / total_combinations:.1f}s")
                
            except Exception as e:
                print(f"  ❌ Parameter sweep failed: {e}")
                raise
            
            # Validate sweep results
            print("  🔍 Validating parameter sweep results...")
            assert len(df) > 0, "Should produce sweep results"
            
            # Check we have all parameter combinations
            param_combinations = df[["drop_left", "drop_right"]].drop_duplicates()
            actual_combinations = len(param_combinations)
            assert actual_combinations == total_combinations, f"Expected {total_combinations} combinations, got {actual_combinations}"
            
            # Check all words are present for each combination
            words_per_combination = df.groupby(["drop_left", "drop_right"])["word"].nunique()
            assert all(words_per_combination == len(words)), "Should have all words for each parameter combination"
            
            # Check output files were created
            csv_files = list(temp_path.glob("entropy_*.csv"))
            assert len(csv_files) == total_combinations, f"Expected {total_combinations} CSV files, got {len(csv_files)}"
            
            # Verify combined results file
            combined_file = temp_path / "combined_results.csv"
            assert combined_file.exists(), "Combined results file should exist"
            
            # Verify metadata file
            metadata_file = temp_path / "metadata.json"
            assert metadata_file.exists(), "Metadata file should exist"
            
            # Show final statistics
            total_rows = len(df)
            unique_words = df["word"].nunique()
            print(f"  📈 Results summary:")
            print(f"    • Total rows: {total_rows:,}")
            print(f"    • Unique words: {unique_words}")
            print(f"    • Parameter combinations: {actual_combinations}")
            print(f"    • Output files: {len(csv_files)}")
            
            print(f"  ✓ Created {len(csv_files)} output files")
            print("  🎉 Parameter sweep test passed!")
    
    @pytest.mark.slow
    def test_full_1k_words_performance(self):
        """Full performance test with 1000 words (marked as slow)."""
        print("\n⚡ Running full 1000-word performance test...")
        
        # Load top 1000 words
        print("  📚 Loading full corpus data...")
        words = get_top_words(1000, word_lengths=range(4, 9))
        print(f"  ✓ Loaded {len(words)} words")
        
        # Full batch processing test with progress
        print("  🧮 Processing all 1000 words...")
        start_time = time.time()
        df = batch_ilp_entropy(
            words,
            drop_left=0.15,
            drop_right=0.25,
            n_workers=8,  # Use more workers for large dataset
            progress_callback=progress_reporter
        )
        total_time = time.time() - start_time
        
        rate = len(words) / total_time
        print(f"  ✓ Processed {len(words)} words in {total_time:.1f}s")
        print(f"  ✓ Processing rate: {rate:.1f} words/sec")
        
        # Performance benchmarks
        assert rate > 0.5, f"Performance too slow: {rate:.2f} words/sec"
        
        # Validate comprehensive results
        assert len(df) > len(words), "Should have multiple positions per word"
        
        unique_words = df["word"].nunique()
        assert unique_words == len(words), f"Expected {len(words)} unique words, got {unique_words}"
        
        print("  🎉 Full 1000-word performance test passed!")


def main():
    """Run performance tests directly."""
    import sys
    
    test_runner = TestPerformance()
    
    try:
        print("🧪 Running ILP Entropy Performance Tests")
        print("=" * 50)
        
        # Run main 1k words test
        test_runner.test_top_1k_words_entropy()
        
        # Run parameter sweep test
        test_runner.test_parameter_sweep_medium_scale()
        
        # Ask if user wants to run slow test
        if "--full" in sys.argv or "--slow" in sys.argv:
            test_runner.test_full_1k_words_performance()
        else:
            print("\n💡 Tip: Use --full flag to run complete 1000-word performance test")
        
        print("\n🎉 All performance tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        import traceback
        print("🔍 Full traceback:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main()) 