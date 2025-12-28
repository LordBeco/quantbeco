#!/usr/bin/env python3
"""
Quick start script for local development
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def run_streamlit():
    """Run the Streamlit app"""
    print("🚀 Starting Trading Performance Intelligence...")
    print("📍 App will open at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the app")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to start Streamlit app")
        return False
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
        return True

def main():
    """Main function"""
    print("🎯 Trading Performance Intelligence - Local Development")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Run the app
    return run_streamlit()

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)