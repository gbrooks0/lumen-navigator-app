# setup_auth0.py - Easy setup script for Lumen Navigator Auth0 integration
import os
import sys
from pathlib import Path
import shutil

def create_secrets_template():
    """Create secrets.toml template"""
    secrets_dir = Path(".streamlit")
    secrets_dir.mkdir(exist_ok=True)
    
    secrets_file = secrets_dir / "secrets.toml"
    
    template = """# Lumen Navigator Auth0 Configuration
# Get these values from your Auth0 Dashboard → Applications → Your App

[auth0]
AUTH0_DOMAIN = "your-tenant.us.auth0.com"  # Replace with your Auth0 domain
AUTH0_CLIENT_ID = "your-client-id-here"    # Replace with your Client ID
AUTH0_CLIENT_SECRET = "your-client-secret-here"  # Replace with your Client Secret
AUTH0_REDIRECT_URI = "http://localhost:8501"  # Local development URL

# Optional: Add your existing API keys here too
OPENAI_API_KEY = "your-openai-key"
GOOGLE_API_KEY = "your-google-key"

# For production, change AUTH0_REDIRECT_URI to your live domain
# AUTH0_REDIRECT_URI = "https://your-domain.com"
"""
    
    if not secrets_file.exists():
        with open(secrets_file, 'w') as f:
            f.write(template)
        print(f"✅ Created {secrets_file}")
        print("📝 Please edit .streamlit/secrets.toml with your Auth0 credentials")
    else:
        print(f"ℹ️  {secrets_file} already exists")

def backup_existing_files():
    """Backup existing auth files"""
    files_to_backup = ['auth_server.py', 'auth_wrapper.py', 'start.sh']
    backup_dir = Path("auth_backup")
    
    backed_up_files = []
    for file in files_to_backup:
        if Path(file).exists():
            backup_dir.mkdir(exist_ok=True)
            shutil.copy2(file, backup_dir / file)
            backed_up_files.append(file)
    
    if backed_up_files:
        print(f"📦 Backed up {len(backed_up_files)} files to auth_backup/")
        return True
    return False

def create_requirements_update():
    """Create or update requirements.txt with needed packages"""
    requirements_file = Path("requirements.txt")
    
    needed_packages = [
        "streamlit",
        "requests",
        "PyJWT",
        "python-dotenv"
    ]
    
    existing_requirements = []
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            existing_requirements = f.read().splitlines()
    
    # Check which packages are missing
    missing_packages = []
    for package in needed_packages:
        if not any(package.lower() in line.lower() for line in existing_requirements):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Adding {len(missing_packages)} packages to requirements.txt")
        with open(requirements_file, 'a') as f:
            f.write("\n# Auth0 integration requirements\n")
            for package in missing_packages:
                f.write(f"{package}\n")
        print("✅ Updated requirements.txt")
    else:
        print("✅ All required packages already in requirements.txt")

def create_simple_start_script():
    """Create a simple start script"""
    start_script = """#!/bin/bash
# Simple start script for Lumen Navigator

echo "🏠 Starting Lumen Navigator..."
echo "📍 Access at: http://localhost:8501"
echo ""

# Install dependencies if needed
pip install -q -r requirements.txt

# Start Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
"""
    
    with open("start_simple.sh", 'w') as f:
        f.write(start_script)
    
    # Make executable on Unix systems
    if sys.platform != 'win32':
        os.chmod("start_simple.sh", 0o755)
    
    print("✅ Created start_simple.sh - simple startup script")

def show_next_steps():
    """Show what to do next"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE! Next steps:")
    print("="*60)
    
    print("\n1. 🔧 Configure Auth0:")
    print("   • Edit .streamlit/secrets.toml with your Auth0 credentials")
    print("   • Get credentials from: https://manage.auth0.com")
    
    print("\n2. 🏗️ Auth0 Dashboard Settings:")
    print("   • Application Type: Regular Web Application")
    print("   • Allowed Callback URLs: http://localhost:8501")
    print("   • Allowed Logout URLs: http://localhost:8501")
    print("   • Allowed Web Origins: http://localhost:8501")
    
    print("\n3. 🚀 Start the application:")
    print("   • Run: streamlit run app.py")
    print("   • Or: ./start_simple.sh")
    
    print("\n4. ✨ What changed:")
    print("   • No more separate auth server needed")
    print("   • Professional Auth0 hosted login page")
    print("   • Single file authentication")
    print("   • Much simpler setup!")
    
    print("\n📧 Need help? Check the setup guide or contact support")
    print("="*60)

def main():
    """Main setup function"""
    print("🏠 Lumen Navigator - Auth0 Setup Script")
    print("=" * 50)
    
    # Check current directory
    if not Path("app.py").exists():
        print("❌ app.py not found. Please run this script in your project directory.")
        return
    
    print("🔍 Setting up streamlined Auth0 integration...")
    
    # Backup existing auth files
    backup_existing_files()
    
    # Create secrets template
    create_secrets_template()
    
    # Update requirements
    create_requirements_update()
    
    # Create simple start script
    create_simple_start_script()
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()
