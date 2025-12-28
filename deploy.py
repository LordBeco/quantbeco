#!/usr/bin/env python3
"""
Automated deployment script for Trading Performance Intelligence
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_git_installed():
    """Check if git is installed"""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git is not installed. Please install Git first.")
        return False

def main():
    """Main deployment function"""
    print("🚀 Trading Performance Intelligence - Deployment Script")
    print("=" * 60)
    
    # Check prerequisites
    if not check_git_installed():
        return False
    
    # Get user input
    print("\n📋 Setup Information:")
    github_username = input("Enter your GitHub username: ").strip()
    repo_name = input("Enter repository name (default: trading-performance-intelligence): ").strip()
    
    if not repo_name:
        repo_name = "trading-performance-intelligence"
    
    if not github_username:
        print("❌ GitHub username is required")
        return False
    
    print(f"\n🎯 Repository will be: https://github.com/{github_username}/{repo_name}")
    confirm = input("Continue? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("❌ Deployment cancelled")
        return False
    
    # Initialize git repository
    if not os.path.exists('.git'):
        if not run_command("git init", "Initializing Git repository"):
            return False
    
    # Add all files
    if not run_command("git add .", "Adding files to Git"):
        return False
    
    # Create initial commit
    if not run_command('git commit -m "Initial commit: Trading Performance Intelligence Dashboard"', "Creating initial commit"):
        # Check if there are changes to commit
        result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
        if not result.stdout.strip():
            print("ℹ️ No changes to commit")
        else:
            return False
    
    # Add remote origin
    remote_url = f"https://github.com/{github_username}/{repo_name}.git"
    run_command(f"git remote remove origin", "Removing existing origin (if any)")  # Don't check result
    if not run_command(f"git remote add origin {remote_url}", "Adding GitHub remote"):
        return False
    
    # Set main branch
    if not run_command("git branch -M main", "Setting main branch"):
        return False
    
    print("\n🌐 Next Steps:")
    print("1. Create a new repository on GitHub:")
    print(f"   - Go to https://github.com/new")
    print(f"   - Repository name: {repo_name}")
    print(f"   - Description: Professional trading analytics dashboard")
    print(f"   - Set to Public")
    print(f"   - Don't initialize with README")
    print(f"   - Click 'Create Repository'")
    
    input("\nPress Enter after creating the GitHub repository...")
    
    # Push to GitHub
    if not run_command("git push -u origin main", "Pushing to GitHub"):
        print("\n💡 If this fails, make sure you:")
        print("   1. Created the repository on GitHub")
        print("   2. Have the correct repository name")
        print("   3. Have push access to the repository")
        return False
    
    print("\n🎉 SUCCESS! Your code is now on GitHub!")
    print(f"📍 Repository URL: https://github.com/{github_username}/{repo_name}")
    
    print("\n🚀 Deploy to Streamlit Cloud:")
    print("1. Go to https://share.streamlit.io")
    print("2. Sign in with GitHub")
    print("3. Click 'New App'")
    print(f"4. Select repository: {github_username}/{repo_name}")
    print("5. Main file path: app.py")
    print("6. Click 'Deploy'")
    
    print(f"\n🌐 Your app will be available at:")
    print(f"https://{github_username}-{repo_name.replace('_', '-')}-app-xxxxx.streamlit.app")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Deployment preparation completed successfully!")
        else:
            print("\n❌ Deployment preparation failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)