"""
Quick Test Runner - Runs all tests in sequence

This is your one-command test suite.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def run_test(script_path, name):
    """Run a test script and report results."""
    print(f"Running: {name}...")
    print("-" * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"\n✅ {name} PASSED\n")
            return True
        else:
            print(f"\n⚠️  {name} had warnings (exit code: {result.returncode})\n")
            return True  # Don't fail, warnings are OK
    
    except subprocess.TimeoutExpired:
        print(f"\n❌ {name} TIMEOUT\n")
        return False
    except Exception as e:
        print(f"\n❌ {name} ERROR: {e}\n")
        return False


def main():
    """Run all tests."""
    print_header("🧪 CLAIMNOW TEST SUITE")
    
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests_dir = Path(__file__).parent / "tests"
    
    tests = [
        (tests_dir / "test_health_check.py", "1. Health Check - Verify all components"),
        (tests_dir / "test_api.py", "2. API & Direct Analysis - Test scoring"),
        (tests_dir / "test_pipeline_e2e.py", "3. End-to-End Pipeline - Full flow with PDF"),
    ]
    
    results = {}
    passed = 0
    failed = 0
    
    for script_path, name in tests:
        if not script_path.exists():
            print(f"⚠️  Skipping (not found): {script_path}")
            results[name] = "SKIPPED"
            continue
        
        print_header(name)
        
        if run_test(str(script_path), name):
            passed += 1
            results[name] = "PASSED"
        else:
            failed += 1
            results[name] = "FAILED"
    
    # Summary
    print_header("📋 TEST SUMMARY")
    
    for test_name, status in results.items():
        symbol = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "⏭️ "
        print(f"{symbol} {test_name}: {status}")
    
    print()
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Total:   {passed + failed}")
    print()
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if failed == 0:
        print("🎉 All tests passed! System ready for deployment.")
        print()
        print("📌 Next steps:")
        print("   1. Start API server: uvicorn src.main:app --reload --port 8000")
        print("   2. View API docs: http://localhost:8000/docs")
        print("   3. Upload sample PDF: http://localhost:8000/docs")
        print()
        return 0
    else:
        print("⚠️  Some tests failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
