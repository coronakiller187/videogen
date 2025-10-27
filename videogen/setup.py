#!/usr/bin/env python3
"""
VideoGen Setup Script
Automated setup for VideoGen in cloud GPU environments
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


class VideoGenSetup:
    """Automated setup for VideoGen."""
    
    def __init__(self):
        """Initialize setup."""
        self.root_dir = Path(__file__).parent
        self.install_dir = self.root_dir
        self.venv_name = "videogen_env"
        
    def check_system_requirements(self):
        """Check system requirements."""
        print("🔍 Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            print("❌ Python 3.8+ required")
            return False
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check for GPU (optional)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                print("⚠️ No GPU detected - generation will be slow")
        except ImportError:
            print("⚠️ PyTorch not installed yet - will be installed with dependencies")
        
        return True
    
    def setup_virtual_environment(self):
        """Create and activate virtual environment."""
        print("🔧 Setting up virtual environment...")
        
        venv_path = self.install_dir / self.venv_name
        
        # Create virtual environment
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            print(f"✅ Virtual environment created at {venv_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
        
        # Determine activation script
        if os.name == 'nt':  # Windows
            activate_script = venv_path / "Scripts" / "activate.bat"
            python_exe = venv_path / "Scripts" / "python.exe"
        else:  # Unix/Linux/macOS
            activate_script = venv_path / "bin" / "activate"
            python_exe = venv_path / "bin" / "python"
        
        # Return paths for later use
        self.venv_path = venv_path
        self.python_exe = python_exe
        self.activate_script = activate_script
        
        return True
    
    def install_dependencies(self):
        """Install Python dependencies."""
        print("📦 Installing dependencies...")
        
        # Upgrade pip first
        subprocess.run([str(self.python_exe), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        requirements_file = self.install_dir / "requirements.txt"
        if requirements_file.exists():
            subprocess.run([str(self.python_exe), "-m", "pip", "install", "-r", str(requirements_file)], check=True)
            print("✅ Dependencies installed successfully")
        else:
            print("❌ requirements.txt not found")
            return False
        
        return True
    
    def setup_directories(self):
        """Create necessary directories."""
        print("📁 Creating directories...")
        
        directories = [
            "outputs",
            "outputs/cinematic",
            "outputs/product",
            "outputs/explainer",
            "outputs/series",
            "models",
            "data",
            "logs"
        ]
        
        for directory in directories:
            dir_path = self.install_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("✅ Directories created")
        return True
    
    def download_models(self):
        """Download pre-trained models (placeholder)."""
        print("🤖 Setting up models...")
        
        # This would download and cache models
        # For now, just create the models directory structure
        models_dir = self.install_dir / "models"
        
        model_subdirs = [
            "videogen_base",
            "lora_models", 
            "controlnet_models",
            "style_models"
        ]
        
        for subdir in model_subdirs:
            (models_dir / subdir).mkdir(exist_ok=True)
        
        print("✅ Model directories created")
        print("ℹ️ Models will be downloaded automatically when first used")
        return True
    
    def create_launch_scripts(self):
        """Create launch scripts for different interfaces."""
        print("🚀 Creating launch scripts...")
        
        # Create scripts directory
        scripts_dir = self.install_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Gradio launcher
        gradio_script = scripts_dir / "launch_gradio.py"
        gradio_script.write_text(f'''#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from interfaces.gradio_ui import VideoGenUI

if __name__ == "__main__":
    ui = VideoGenUI()
    ui.launch(share=False, debug=False)
''')
        
        # CLI launcher
        cli_script = scripts_dir / "videogen_cli.py"
        cli_script.write_text(f'''#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from interfaces.cli import main

if __name__ == "__main__":
    main()
''')
        
        # Make scripts executable
        if os.name != 'nt':  # Unix/Linux/macOS
            os.chmod(gradio_script, 0o755)
            os.chmod(cli_script, 0o755)
        
        print("✅ Launch scripts created")
        return True
    
    def create_config_file(self):
        """Create default configuration file."""
        print("⚙️ Creating configuration file...")
        
        config_content = '''# VideoGen Configuration
# Cloud-optimized settings

# Core settings
resolution: 512x512
fps: 8
duration: 5.0
guidance_scale: 7.5
inference_steps: 20

# Cloud GPU optimizations
memory_efficient: true
fp16: true
batch_size: 1
gradient_checkpointing: true
max_memory_fraction: 0.9

# Quality settings
quality_mode: balanced
noise_reduction: true
motion_smoothing: true

# Output settings
output_format: mp4
compress_output: true

# Logging
verbose: true
save_metadata: true
monitor_memory: true
memory_threshold: 0.85
'''
        
        config_file = self.install_dir / "config.yaml"
        config_file.write_text(config_content)
        
        print("✅ Configuration file created")
        return True
    
    def setup_colab_notebook(self):
        """Setup Colab-specific files."""
        print("📓 Setting up Colab integration...")
        
        # Copy Colab notebook
        source_notebook = self.install_dir / "examples" / "notebooks" / "VideoGen_Complete_Setup.ipynb"
        target_notebook = self.install_dir / "VideoGen_Colab_Setup.ipynb"
        
        if source_notebook.exists():
            target_notebook.write_text(source_notebook.read_text())
            print("✅ Colab notebook ready")
        else:
            print("⚠️ Colab notebook not found")
        
        return True
    
    def run_setup(self):
        """Run the complete setup process."""
        print("🎬 VideoGen Setup")
        print("=" * 50)
        
        steps = [
            ("Checking system requirements", self.check_system_requirements),
            ("Setting up virtual environment", self.setup_virtual_environment),
            ("Installing dependencies", self.install_dependencies),
            ("Creating directories", self.setup_directories),
            ("Setting up models", self.download_models),
            ("Creating launch scripts", self.create_launch_scripts),
            ("Creating configuration", self.create_config_file),
            ("Setting up Colab", self.setup_colab_notebook)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            if not step_func():
                print(f"❌ Setup failed at: {step_name}")
                return False
            
            time.sleep(0.5)  # Small delay for readability
        
        print("\n" + "=" * 50)
        print("🎉 VideoGen setup completed successfully!")
        print("\n📚 Next Steps:")
        print("1. Activate the virtual environment:")
        if os.name == 'nt':
            print(f"   {self.venv_path}\\Scripts\\activate")
        else:
            print(f"   source {self.venv_path}/bin/activate")
        
        print("\n2. Launch the web interface:")
        print(f"   python {self.install_dir}/scripts/launch_gradio.py")
        
        print("\n3. Or use the command line:")
        print(f"   python {self.install_dir}/scripts/videogen_cli.py --help")
        
        print("\n4. For Colab, upload and run:")
        print(f"   {self.install_dir}/VideoGen_Colab_Setup.ipynb")
        
        return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="VideoGen Setup Script")
    parser.add_argument("--install-dir", default=None, help="Installation directory")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--force", action="store_true", help="Force setup even if errors occur")
    
    args = parser.parse_args()
    
    # Change to install directory if specified
    if args.install_dir:
        os.chdir(args.install_dir)
    
    # Run setup
    setup = VideoGenSetup()
    
    try:
        success = setup.run_setup()
        if not success and not args.force:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        if not args.force:
            sys.exit(1)


if __name__ == "__main__":
    main()