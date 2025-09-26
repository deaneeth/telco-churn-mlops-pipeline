#!/usr/bin/env python3
"""
Test execution script for Telco Churn Prediction project

This script provides convenient ways to run different types of tests
with appropriate configurations and reporting.

Usage:
    python run_tests.py [OPTIONS]
    
Examples:
    python run_tests.py --all              # Run all tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --performance      # Run performance tests
    python run_tests.py --quick            # Run smoke tests only
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return the result"""
    if description:
        print(f"\n🔄 {description}")
        print("-" * 50)
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return 1

def main():
    parser = argparse.ArgumentParser(description="Run tests for Telco Churn Prediction project")
    
    # Test selection options
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--api", action="store_true", help="Run API tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke tests only")
    
    # Reporting options
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")
    
    # Advanced options
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--profile", action="store_true", help="Profile test execution")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests")
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Test selection
    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])
    elif args.api:
        cmd.extend(["-m", "api"])
    elif args.performance:
        cmd.extend(["-m", "performance"])
    elif args.quick:
        cmd.extend(["-m", "smoke", "--maxfail=1"])
    elif not args.all:
        # Default: run unit and integration tests
        cmd.extend(["-m", "unit or integration"])
    
    # Coverage options
    if args.coverage or args.all:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=xml:coverage.xml",
            "--cov-fail-under=85"
        ])
    
    # HTML reporting
    if args.html or args.all:
        cmd.extend([
            "--cov-report=html:htmlcov",
            "--html=test-report.html",
            "--self-contained-html"
        ])
    
    # Output options
    if args.verbose:
        cmd.append("-v")
    elif args.quiet:
        cmd.append("-q")
    
    # Parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    # Profiling
    if args.profile:
        cmd.extend(["--durations=10"])
    
    # Benchmarking
    if args.benchmark:
        cmd.extend(["-m", "benchmark", "--benchmark-only"])
    
    # Add standard options
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "--junit-xml=test-results.xml"
    ])
    
    # Check if we're in the right directory
    if not Path("tests").exists():
        print("❌ Tests directory not found. Please run from project root.")
        return 1
    
    # Install test dependencies if needed
    if not Path(".pytest_cache").exists():
        print("📦 Installing test dependencies...")
        install_cmd = ["pip", "install", "-r", "test-requirements.txt"]
        install_result = run_command(install_cmd, "Installing test dependencies")
        if install_result != 0:
            print("⚠️  Warning: Failed to install test dependencies. Some tests might fail.")
    
    # Run the tests
    print("\n🚀 RUNNING TELCO CHURN PREDICTION TESTS")
    print("=" * 60)
    
    exit_code = run_command(cmd, "Executing test suite")
    
    # Summary
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        
        # Display coverage info if generated
        if args.coverage or args.all:
            print("\n📊 Coverage report available in htmlcov/index.html")
        
        if args.html or args.all:
            print("📋 Test report available in test-report.html")
            
    else:
        print(f"❌ TESTS FAILED (exit code: {exit_code})")
        print("Check the output above for detailed error information.")
    
    print("=" * 60)
    return exit_code

if __name__ == "__main__":
    sys.exit(main())