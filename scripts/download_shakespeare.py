#!/usr/bin/env python3
"""
Download Shakespeare text for pretraining experiments
"""

import os
import urllib.request

DATA_DIR = "../data/raw"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUTPUT_FILE = os.path.join(DATA_DIR, "shakespeare.txt")

def download_shakespeare():
    """Download Shakespeare dataset"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("Downloading Shakespeare dataset...")
    print(f"URL: {SHAKESPEARE_URL}")
    print(f"Saving to: {OUTPUT_FILE}")
    
    try:
        urllib.request.urlretrieve(SHAKESPEARE_URL, OUTPUT_FILE)
        print("\n✓ Download complete!")
        
        # Show file info
        size = os.path.getsize(OUTPUT_FILE)
        print(f"\nFile size: {size / 1024:.2f} KB ({size / (1024*1024):.2f} MB)")
        
        # Show first few lines
        print("\nFirst few lines:")
        print("-" * 60)
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                print(line.rstrip())
        print("-" * 60)
        
    except Exception as e:
        print(f"\n✗ Error downloading file: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = download_shakespeare()
    exit(0 if success else 1)
