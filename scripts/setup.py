#!/usr/bin/env python3
"""Setup script for the MLOps project."""

import os
import subprocess
import sys
import platform
from pathlib import Path

def run_command(command: str, description: str, check: bool = True, verbose: bool = False) -> bool:
    """Run a shell command and return success status."""
    print(f"🔄 {description}...")
    try:
        if verbose:
            # 실시간 출력을 위해 capture_output=False
            result = subprocess.run(command, shell=True, check=check)
        else:
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            if not verbose:
                print(f"❌ {description} failed: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if not verbose:
            print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_virtual_environment():
    """Create a virtual environment."""
    venv_name = "venv"
    
    if Path(venv_name).exists():
        print(f"📦 Virtual environment '{venv_name}' already exists")
        return True
    
    print(f"📦 Creating virtual environment '{venv_name}'...")
    
    if platform.system() == "Windows":
        # Windows
        if not run_command(f"python -m venv {venv_name}", "Creating virtual environment"):
            return False
        
        # Activate virtual environment
        activate_script = Path(venv_name) / "Scripts" / "activate.bat"
        if activate_script.exists():
            print("📦 Virtual environment created successfully")
            print(f"📋 To activate: {venv_name}\\Scripts\\activate")
            return True
        else:
            print("❌ Virtual environment activation script not found")
            return False
    else:
        # Unix/Linux/macOS
        if not run_command(f"python3 -m venv {venv_name}", "Creating virtual environment"):
            return False
        
        activate_script = Path(venv_name) / "bin" / "activate"
        if activate_script.exists():
            print("📦 Virtual environment created successfully")
            print(f"📋 To activate: source {venv_name}/bin/activate")
            return True
        else:
            print("❌ Virtual environment activation script not found")
            return False

def install_dependencies_in_venv():
    """Install dependencies in virtual environment."""
    print("📦 Installing dependencies in virtual environment...")
    
    # Determine the correct pip path
    if platform.system() == "Windows":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
    
    # Upgrade pip first
    if not run_command(f"{pip_path} install --upgrade pip", "Upgrading pip"):
        print("⚠️  Failed to upgrade pip, continuing...")
    
    # Install main dependencies with verbose output
    print("📦 Installing main dependencies (this may take a while)...")
    print("📋 Installing packages: scikit-learn, pandas, numpy, mlflow, fastapi, etc.")
    if not run_command(f"{pip_path} install -e . --verbose", "Installing main dependencies", verbose=True):
        print("❌ Failed to install main dependencies")
        return False
    
    # Install dev dependencies
    print("📦 Installing development dependencies...")
    if not run_command(f"{pip_path} install -e .[dev] --verbose", "Installing dev dependencies", verbose=True):
        print("⚠️  Failed to install dev dependencies, continuing...")
    
    return True

def create_directories():
    """Create necessary project directories."""
    print("📁 Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/features",
        "models",
        "logs",
        "notebooks",
        "docs",
        "monitoring/grafana/dashboards",
        "monitoring/grafana/datasources",
        "nginx",
        "infrastructure"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def setup_dvc():
    """Initialize DVC if not already initialized."""
    if not Path(".dvc").exists():
        # Use the virtual environment's dvc
        if platform.system() == "Windows":
            dvc_path = "venv\\Scripts\\dvc"
        else:
            dvc_path = "venv/bin/dvc"
        
        if not run_command(f"{dvc_path} init", "Initializing DVC", check=False):
            print("⚠️  DVC initialization failed, continuing..")
        else:
            print("📊 DVC initialized successfully")
    else:
        print("📊 DVC already initialized")

def setup_pre_commit():
    """Install pre-commit hooks."""
    if platform.system() == "Windows":
        pre_commit_path = "venv\\Scripts\\pre-commit"
    else:
        pre_commit_path = "venv/bin/pre-commit"
    
    if run_command(f"{pre_commit_path} install", "Installing pre-commit hooks", check=False):
        print("✅ Pre-commit hooks installed")
    else:
        print("⚠️  Pre-commit installation failed, continuing..")

def setup_environment():
    """Copy env.example to .env if .env doesn't exist."""
    if not Path(".env").exists():
        if Path("env.example").exists():
            import shutil
            shutil.copy("env.example", ".env")
            print("📝 Created .env file from env.example")
        else:
            print("⚠️  env.example not found")
    else:
        print("📝 .env file already exists")

def setup_git():
    """Initialize git repository if not already initialized."""
    if not Path(".git").exists():
        if run_command("git init", "Initializing git repository"):
            run_command("git add .", "Adding files to git")
            run_command('git commit -m "Initial commit"', "Making initial commit")
        else:
            print("⚠️  Git initialization failed")
    else:
        print("📝 Git repository already exists")

def activate_virtual_environment():
    """Activate the virtual environment and return the activated environment."""
    if platform.system() == "Windows":
        activate_script = Path("venv") / "Scripts" / "activate.bat"
        if activate_script.exists():
            print("🔄 Activating virtual environment...")
            # Windows에서는 환경 변수를 설정하여 가상환경 활성화 효과를 시뮬레이션
            os.environ["VIRTUAL_ENV"] = str(Path("venv").absolute())
            os.environ["PATH"] = str(Path("venv") / "Scripts") + os.pathsep + os.environ["PATH"]
            print("✅ Virtual environment activated")
            return True
        else:
            print("❌ Virtual environment activation script not found")
            return False
    else:
        activate_script = Path("venv") / "bin" / "activate"
        if activate_script.exists():
            print("🔄 Activating virtual environment...")
            # Unix에서는 환경 변수를 설정
            os.environ["VIRTUAL_ENV"] = str(Path("venv").absolute())
            os.environ["PATH"] = str(Path("venv") / "bin") + os.pathsep + os.environ["PATH"]
            print("✅ Virtual environment activated")
            return True
        else:
            print("❌ Virtual environment activation script not found")
            return False

def print_activation_instructions():
    """Print instructions for activating the virtual environment."""
    print("\n" + "="*60)
    print("🎯 VIRTUAL ENVIRONMENT ACTIVATION")
    print("="*60)
    
    if platform.system() == "Windows":
        print("📋 To activate the virtual environment:")
        print("   venv\\Scripts\\activate")
        print("\n📋 Or in PowerShell:")
        print("   venv\\Scripts\\Activate.ps1")
    else:
        print("📋 To activate the virtual environment:")
        print("   source venv/bin/activate")
    
    print("\n📋 After activation, you can run:")
    print("   python -m src.api.main")
    print("   docker-compose up -d")

def main():
    print("🚀 Setting up MLOps Lifecycle Management Project")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Failed to create virtual environment")
        return
    
    # Create directories
    create_directories()
    
    # Setup git
    setup_git()
    
    # Install dependencies in virtual environment
    if not install_dependencies_in_venv():
        print("❌ Setup failed due to dependency installation issues")
        return
    
    # Setup DVC
    setup_dvc()
    
    # Setup environment
    setup_environment()
    
    # Setup pre-commit (optional)
    setup_pre_commit()
    
    # Activate virtual environment
    if activate_virtual_environment():
        print("\n✅ Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Virtual environment is now activated!")
        print("2. Edit .env file with your configuration")
        print("3. Run 'docker-compose up -d' to start infrastructure services")
        print("4. Run 'python -m src.api.main' to start the API server")
        print("5. Visit http://localhost:8000/docs for API documentation")
        print("6. Visit http://localhost:3000 for Grafana dashboard")
        print("7. Visit http://localhost:5000 for MLflow tracking")
        
        # 자동으로 다음 단계 실행 여부 확인
        print("\n🚀 Would you like to start the services now? (y/n): ", end="")
        try:
            response = input().lower().strip()
            if response in ['y', 'yes', 'ㅇ']:
                # Docker 체크
                print("\n🔍 Checking Docker...")
                if not run_command("docker --version", "Checking Docker installation", check=False):
                    print("⚠️  Docker not found or not running")
                    print("📋 Please start Docker Desktop and try again")
                    print("📋 Manual commands:")
                    print("   docker-compose up -d")
                    print("   python -m src.api.main")
                    return
                
                print("\n🔄 Starting infrastructure services...")
                if run_command("docker-compose up -d", "Starting Docker services", verbose=True):
                    print("\n🔄 Starting API server...")
                    run_command("python -m src.api.main", "Starting API server", verbose=True)
                else:
                    print("⚠️  Docker services failed to start")
                    print("📋 Please check Docker Desktop and try again")
            else:
                print("\n📋 Manual commands:")
                print("   docker-compose up -d")
                print("   python -m src.api.main")
        except KeyboardInterrupt:
            print("\n📋 Manual commands:")
            print("   docker-compose up -d")
            print("   python -m src.api.main")
    else:
        print("\n✅ Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Activate the virtual environment manually:")
        print("   venv\\Scripts\\activate")
        print("2. Edit .env file with your configuration")
        print("3. Run 'docker-compose up -d' to start infrastructure services")
        print("4. Run 'python -m src.api.main' to start the API server")
        print("5. Visit http://localhost:8000/docs for API documentation")
        print("6. Visit http://localhost:3000 for Grafana dashboard")
        print("7. Visit http://localhost:5000 for MLflow tracking")
        
        print_activation_instructions()

if __name__ == "__main__":
    main() 