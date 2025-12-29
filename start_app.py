#!/usr/bin/env python3
"""
Simple startup script for Trade Analyzer Pro with Calendar Feature
"""

import subprocess
import sys
import os

def main():
    print("🎯 Starting Trade Analyzer Pro with Calendar Feature...")
    print("📅 New Feature: Daily Trading Calendar with color-coded P&L!")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found. Please run this script from the trade_analyzer_pro directory.")
        return
    
    # Clear any cached modules
    if os.path.exists('__pycache__'):
        print("🧹 Clearing Python cache...")
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['rmdir', '/s', '/q', '__pycache__'], shell=True, check=False)
            else:  # Unix/Linux/Mac
                subprocess.run(['rm', '-rf', '__pycache__'], check=False)
        except:
            pass
    
    print("🚀 Launching Streamlit application...")
    print("📊 The calendar chart will appear in the 'Time Analysis' section")
    print("🌐 Your browser should open automatically at http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Start Streamlit
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting application: {e}")
        print("💡 Try running: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()