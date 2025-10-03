"""
Automated MLflow UI screenshot capture using Playwright.
Captures runs list and model artifact pages.
"""
import time
import sys
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
    
    print("🎬 Starting automated screenshot capture...")
    
    with sync_playwright() as p:
        # Launch browser
        print("  → Launching browser...")
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1920, 'height': 1080})
        
        # Screenshot 1: MLflow runs list (landing page)
        print("  → Navigating to MLflow UI (http://127.0.0.1:5000)...")
        try:
            page.goto("http://127.0.0.1:5000", timeout=15000, wait_until="networkidle")
            time.sleep(2)  # Extra wait for dynamic content
            
            screenshot_path = Path("docs/images/mlflow_runs.png")
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"  ✓ Screenshot 1 saved: {screenshot_path}")
            
        except Exception as e:
            print(f"  ✗ Failed to capture runs list: {e}")
            
        # Screenshot 2: Model artifact page
        print("  → Navigating to latest model run...")
        try:
            # Try to find and click on the first run in the table
            page.wait_for_selector("table", timeout=5000)
            
            # Click on first run row
            first_run = page.locator("table tbody tr").first
            if first_run.count() > 0:
                first_run.click()
                time.sleep(2)
                
                # Navigate to artifacts tab
                artifacts_tab = page.locator("text=Artifacts")
                if artifacts_tab.count() > 0:
                    artifacts_tab.click()
                    time.sleep(2)
                    
                    screenshot_path = Path("docs/images/mlflow_model.png")
                    page.screenshot(path=str(screenshot_path), full_page=True)
                    print(f"  ✓ Screenshot 2 saved: {screenshot_path}")
                else:
                    print("  ⚠ Artifacts tab not found, taking current page screenshot...")
                    screenshot_path = Path("docs/images/mlflow_model.png")
                    page.screenshot(path=str(screenshot_path), full_page=True)
                    print(f"  ✓ Screenshot 2 saved: {screenshot_path}")
            else:
                print("  ⚠ No runs found in table")
                
        except Exception as e:
            print(f"  ⚠ Could not navigate to model page: {e}")
            # Take screenshot of current state anyway
            screenshot_path = Path("docs/images/mlflow_model.png")
            page.screenshot(path=str(screenshot_path), full_page=True)
            print(f"  ✓ Screenshot 2 saved (fallback): {screenshot_path}")
        
        browser.close()
        print("\n✅ Screenshot capture completed!")
        sys.exit(0)
        
except ImportError:
    print("✗ Playwright not available. Creating placeholder files...")
    
    # Create placeholder files
    placeholder_msg = "PLACEHOLDER: Please capture this screenshot manually from http://localhost:5000\n"
    
    Path("docs/images").mkdir(parents=True, exist_ok=True)
    Path("docs/images/mlflow_runs.png.placeholder").write_text(
        placeholder_msg + "Screenshot 1: MLflow runs list (landing page)"
    )
    Path("docs/images/mlflow_model.png.placeholder").write_text(
        placeholder_msg + "Screenshot 2: Model artifact page (click on a run, then Artifacts tab)"
    )
    
    print("  → Placeholder files created:")
    print("     - docs/images/mlflow_runs.png.placeholder")
    print("     - docs/images/mlflow_model.png.placeholder")
    print("\n📝 Manual capture instructions:")
    print("   1. Open http://localhost:5000 in your browser")
    print("   2. Take screenshot of runs list page → save as docs/images/mlflow_runs.png")
    print("   3. Click on any run, go to Artifacts tab → save as docs/images/mlflow_model.png")
    
    sys.exit(1)
    
except Exception as e:
    print(f"✗ Screenshot capture failed: {e}")
    sys.exit(1)
