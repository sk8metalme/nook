#!/usr/bin/env python3
"""
Test execution script for Nook Claude migration tests.

This script provides easy execution of different test categories
following the TDD approach outlined in the test design document.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description="Running command"):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def run_unit_tests():
    """Run unit tests for Claude client."""
    return run_command([
        'python', '-m', 'pytest',
        'nook/functions/common/python/tests/',
        '-v', '--tb=short'
    ], "Running Claude Client Unit Tests")


def run_integration_tests():
    """Run integration tests."""
    success = True

    # Run basic Claude integration tests
    success &= run_command([
        'python', '-m', 'pytest',
        'tests/integration/test_claude_basic_integration.py',
        '-v', '--tb=short'
    ], "Running Basic Claude Integration Tests")

    return success


def run_rollback_tests():
    """Run rollback and environment switching tests."""
    return run_command([
        'python', '-m', 'pytest',
        'tests/integration/test_basic_rollback.py',
        '-v', '--tb=short'
    ], "Running Rollback and Environment Switching Tests")


def run_smoke_tests():
    """Run smoke tests."""
    return run_command([
        'python', '-m', 'pytest',
        'tests/integration/test_basic_rollback.py::TestBasicSmokeTests',
        '-v', '--tb=short'
    ], "Running Smoke Tests")


def run_all_tests():
    """Run all available tests."""
    print("\n" + "="*80)
    print("RUNNING COMPLETE NOOK CLAUDE MIGRATION TEST SUITE")
    print("="*80)

    results = []

    # Unit Tests
    print("\n" + "🧪 PHASE 1: UNIT TESTS")
    results.append(("Unit Tests", run_unit_tests()))

    # Integration Tests
    print("\n" + "🔗 PHASE 2: INTEGRATION TESTS")
    results.append(("Integration Tests", run_integration_tests()))

    # Rollback Tests
    print("\n" + "🔄 PHASE 3: ROLLBACK TESTS")
    results.append(("Rollback Tests", run_rollback_tests()))

    # Smoke Tests
    print("\n" + "💨 PHASE 4: SMOKE TESTS")
    results.append(("Smoke Tests", run_smoke_tests()))

    # Summary
    print("\n" + "="*80)
    print("TEST EXECUTION SUMMARY")
    print("="*80)

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<25} {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} test categories passed")

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Migration test suite is ready.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test categories failed. Review above output.")
        return False


def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        'pytest',
        'pytest-mock',
        'anthropic'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)

    if missing:
        print(f"⚠️  Missing required packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements-test.txt")
        return False

    return True


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description='Run Nook Claude migration tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--rollback', action='store_true', help='Run rollback tests only')
    parser.add_argument('--smoke', action='store_true', help='Run smoke tests only')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies')

    args = parser.parse_args()

    # Check dependencies if requested
    if args.check_deps:
        if check_dependencies():
            print("✅ All dependencies are available")
            return 0
        else:
            return 1

    # Verify we're in the right directory
    if not Path('nook/functions/common/python/claude_client.py').exists():
        print("❌ Error: Please run this script from the project root directory")
        return 1

    success = True

    # Run specific test categories
    if args.unit:
        success = run_unit_tests()
    elif args.integration:
        success = run_integration_tests()
    elif args.rollback:
        success = run_rollback_tests()
    elif args.smoke:
        success = run_smoke_tests()
    else:
        # Run all tests by default
        success = run_all_tests()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())