"""
Automated Screenshot Capture Script for Customer Intent Discovery Platform.
Uses Playwright to capture authentic high-resolution UI screenshots for README.md.
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path(__file__).resolve().parent / "docs" / "screenshots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def capture_all():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 960})
        
        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle")
        time.sleep(3)
        
        # 1. Landing / Data Ingestion Screenshot
        print("Capturing 01_data_ingestion.png...")
        page.screenshot(path=str(OUTPUT_DIR / "01_data_ingestion.png"))
        
        # Select "Load Benchmark Sample" in the sidebar
        print("Selecting Benchmark Dataset...")
        page.locator("text=Load Benchmark Sample").click()
        time.sleep(1)
        
        # Click "Load Selected Benchmark"
        page.locator("button:has-text('Load Selected Benchmark')").click()
        time.sleep(2)
        
        # Click "Apply Preprocessing & Prepare Embeddings"
        print("Applying Preprocessing...")
        page.locator("button:has-text('Apply Preprocessing & Prepare Embeddings')").click()
        time.sleep(3)
        
        # 2. Dataset Profiling & Audit Screenshot
        print("Capturing 02_dataset_profiling.png...")
        page.screenshot(path=str(OUTPUT_DIR / "02_dataset_profiling.png"))
        
        # Tab 1: Discovery Engine (index 1)
        print("Clicking Tab 2: Discovery Engine...")
        page.get_by_role("tab").nth(1).click()
        time.sleep(2)
        
        # Click "Discover Customer Intents Now"
        print("Triggering Intent Discovery...")
        page.locator("button:has-text('Discover Customer Intents Now')").click()
        
        # Wait for discovery to complete
        print("Waiting for clustering and evaluation to complete...")
        page.wait_for_selector("text=Intent discovery complete", timeout=120000)
        time.sleep(3)
        
        # 3. Discovery Engine Completed Screenshot
        print("Capturing 03_discovery_engine.png...")
        page.screenshot(path=str(OUTPUT_DIR / "03_discovery_engine.png"))
        
        # Tab 2: Discovered Intents (index 2)
        print("Clicking Tab 3: Discovered Intents...")
        page.get_by_role("tab").nth(2).click()
        time.sleep(3)
        print("Capturing 04_discovered_intents.png...")
        page.screenshot(path=str(OUTPUT_DIR / "04_discovered_intents.png"))
        
        # Tab 3: Visualizations (index 3)
        print("Clicking Tab 4: Visualizations...")
        page.get_by_role("tab").nth(3).click()
        time.sleep(4)
        print("Capturing 05_semantic_map_2d.png...")
        page.screenshot(path=str(OUTPUT_DIR / "05_semantic_map_2d.png"))
        
        # Tab 4: Quality Evaluation (index 4)
        print("Clicking Tab 5: Quality Evaluation...")
        page.get_by_role("tab").nth(4).click()
        time.sleep(3)
        print("Capturing 06_quality_evaluation.png...")
        page.screenshot(path=str(OUTPUT_DIR / "06_quality_evaluation.png"))
        
        # Tab 5: Export Center (index 5)
        print("Clicking Tab 6: Export Center...")
        page.get_by_role("tab").nth(5).click()
        time.sleep(2)
        print("Capturing 07_export_center.png...")
        page.screenshot(path=str(OUTPUT_DIR / "07_export_center.png"))
        
        browser.close()
        print("SUCCESS: All 7 screenshots captured!")


if __name__ == "__main__":
    capture_all()
