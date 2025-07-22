"""
Environment Setup Script for LLM Optimization Toolkit

This script helps users set up their environment correctly for the LLM optimization toolkit.
It checks system requirements, installs dependencies, and configures the environment.

Features:
- System compatibility check
- Automatic dependency installation
- GPU/CUDA setup verification
- Environment configuration
- Troubleshooting assistance
"""

import sys
import subprocess
import platform
import importlib
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

class EnvironmentSetup:
    """Comprehensive environment setup for LLM optimization toolkit."""
    
    def __init__(self):
        """Initialize the setup process."""
        self.system_info = self._get_system_info()
        self.requirements = self._load_requirements()
        self.setup_log = []
        
    def _get_system_info(self) -> Dict:
        """Get comprehensive system information."""
        info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.architecture()[0],
            "python_version": sys.version,
            "python_executable": sys.executable,
        }
        
        # Check for CUDA availability
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            else:
                info["cuda_version"] = None
                info["gpu_count"] = 0
                info["gpu_names"] = []
        except ImportError:
            info["torch_version"] = None
            info["cuda_available"] = False
            info["cuda_version"] = None
            info["gpu_count"] = 0
            info["gpu_names"] = []
        
        return info
    
    def _load_requirements(self) -> Dict:
        """Load requirements from requirements.txt."""
        requirements_file = Path("requirements.txt")
        
        if not requirements_file.exists():
            print("⚠️  Warning: requirements.txt not found. Using default requirements.")
            return self._get_default_requirements()
        
        requirements = {}
        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '>=' in line:
                        package, version = line.split('>=')
                        requirements[package] = version
                    else:
                        requirements[line] = None
        
        return requirements
    
    def _get_default_requirements(self) -> Dict:
        """Get default requirements if file is not found."""
        return {
            "torch": "2.0.0",
            "transformers": "4.35.0",
            "bitsandbytes": "0.41.0",
            "accelerate": "0.24.0",
            "peft": "0.6.0",
            "datasets": "2.14.0",
            "evaluate": "0.4.0",
            "scikit-learn": "1.0.0",
            "matplotlib": "3.5.0",
            "seaborn": "0.11.0",
            "pandas": "1.3.0",
            "numpy": "1.21.0",
            "tqdm": "4.64.0"
        }
    
    def run_setup(self, mode: str = "interactive") -> bool:
        """
        Run the complete setup process.
        
        Args:
            mode: Setup mode ('interactive', 'auto', 'check-only')
        
        Returns:
            True if setup successful, False otherwise
        """
        print("🚀 LLM Optimization Toolkit - Environment Setup")
        print("=" * 60)
        
        # System compatibility check
        if not self._check_system_compatibility():
            return False
        
        if mode == "check-only":
            return self._check_installation()
        
        # Install dependencies
        if mode == "interactive":
            if not self._interactive_setup():
                return False
        elif mode == "auto":
            if not self._automatic_setup():
                return False
        
        # Verify installation
        if not self._verify_installation():
            return False
        
        # Configure environment
        self._configure_environment()
        
        # Generate setup report
        self._generate_setup_report()
        
        print("\\n✅ Setup completed successfully!")
        print("🎉 You're ready to use the LLM Optimization Toolkit!")
        
        return True
    
    def _check_system_compatibility(self) -> bool:
        """Check if the system meets minimum requirements."""
        print("\\n🔍 Checking system compatibility...")
        
        compatible = True
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            print(f"❌ Python {python_version.major}.{python_version.minor} is too old. Minimum required: 3.8")
            compatible = False
        else:
            print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} - OK")
        
        # Check platform
        if self.system_info["platform"] not in ["Windows", "Linux", "Darwin"]:
            print(f"⚠️  Platform {self.system_info['platform']} may not be fully supported")
        else:
            print(f"✅ Platform {self.system_info['platform']} - OK")
        
        # Check architecture
        if self.system_info["architecture"] != "64bit":
            print(f"⚠️  Architecture {self.system_info['architecture']} may have limited support")
        else:
            print(f"✅ Architecture {self.system_info['architecture']} - OK")
        
        # Check available memory
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < 8:
                print(f"⚠️  Low system memory: {memory_gb:.1f} GB (recommended: 16+ GB)")
            else:
                print(f"✅ System memory: {memory_gb:.1f} GB - OK")
        except ImportError:
            print("ℹ️  Could not check system memory (psutil not installed)")
        
        # Check CUDA availability
        if self.system_info["cuda_available"]:
            print(f"✅ CUDA {self.system_info['cuda_version']} available with {self.system_info['gpu_count']} GPU(s)")
            for i, gpu_name in enumerate(self.system_info["gpu_names"]):
                print(f"   GPU {i}: {gpu_name}")
        else:
            print("ℹ️  CUDA not available - CPU-only mode")
        
        return compatible
    
    def _interactive_setup(self) -> bool:
        """Interactive setup with user prompts."""
        print("\\n🛠️  Interactive Setup")
        print("-" * 30)
        
        # Ask about installation preferences
        print("\\nChoose installation options:")
        print("1. Full installation (all features)")
        print("2. Minimal installation (core features only)")
        print("3. Development installation (includes dev tools)")
        print("4. Custom installation")
        
        while True:
            choice = input("\\nEnter your choice (1-4): ").strip()
            if choice in ["1", "2", "3", "4"]:
                break
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
        
        if choice == "1":
            return self._install_full()
        elif choice == "2":
            return self._install_minimal()
        elif choice == "3":
            return self._install_development()
        else:
            return self._install_custom()
    
    def _automatic_setup(self) -> bool:
        """Automatic setup without user interaction."""
        print("\\n🤖 Automatic Setup")
        print("-" * 20)
        
        # Determine best installation based on system
        if self.system_info["cuda_available"]:
            print("CUDA detected - installing full version with GPU support")
            return self._install_full()
        else:
            print("No CUDA detected - installing CPU version")
            return self._install_minimal()
    
    def _install_full(self) -> bool:
        """Install full version with all features."""
        print("\\n📦 Installing full version...")
        
        packages_to_install = list(self.requirements.keys())
        
        # Add optional packages for full installation
        optional_packages = [
            "jupyter",
            "streamlit",
            "plotly",
            "auto-gptq",
            "awq",
            "flash-attn"
        ]
        
        packages_to_install.extend(optional_packages)
        
        return self._install_packages(packages_to_install)
    
    def _install_minimal(self) -> bool:
        """Install minimal version with core features only."""
        print("\\n📦 Installing minimal version...")
        
        core_packages = [
            "torch",
            "transformers",
            "bitsandbytes",
            "accelerate",
            "numpy",
            "tqdm"
        ]
        
        return self._install_packages(core_packages)
    
    def _install_development(self) -> bool:
        """Install development version with dev tools."""
        print("\\n📦 Installing development version...")
        
        packages_to_install = list(self.requirements.keys())
        
        # Add development packages
        dev_packages = [
            "pytest",
            "black",
            "flake8",
            "mypy",
            "pre-commit",
            "jupyter",
            "streamlit"
        ]
        
        packages_to_install.extend(dev_packages)
        
        return self._install_packages(packages_to_install)
    
    def _install_custom(self) -> bool:
        """Custom installation with user selection."""
        print("\\n📦 Custom Installation")
        print("Select packages to install:")
        
        available_packages = list(self.requirements.keys())
        selected_packages = []
        
        for i, package in enumerate(available_packages, 1):
            print(f"{i:2d}. {package}")
        
        print("\\nEnter package numbers separated by commas (e.g., 1,2,5-8):")
        print("Or enter 'all' to select all packages")
        
        selection = input("Selection: ").strip()
        
        if selection.lower() == 'all':
            selected_packages = available_packages
        else:
            # Parse selection
            try:
                for part in selection.split(','):
                    part = part.strip()
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        selected_packages.extend(available_packages[start-1:end])
                    else:
                        selected_packages.append(available_packages[int(part)-1])
            except (ValueError, IndexError):
                print("Invalid selection. Installing core packages.")
                selected_packages = ["torch", "transformers", "bitsandbytes"]
        
        return self._install_packages(selected_packages)
    
    def _install_packages(self, packages: List[str]) -> bool:
        """Install specified packages."""
        print(f"\\n📥 Installing {len(packages)} packages...")
        
        failed_packages = []
        
        for package in packages:
            print(f"Installing {package}...", end=" ")
            
            try:
                # Determine package specification
                if package in self.requirements and self.requirements[package]:
                    package_spec = f"{package}>={self.requirements[package]}"
                else:
                    package_spec = package
                
                # Install package
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package_spec],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per package
                )
                
                if result.returncode == 0:
                    print("✅")
                    self.setup_log.append(f"Successfully installed {package}")
                else:
                    print("❌")
                    failed_packages.append(package)
                    self.setup_log.append(f"Failed to install {package}: {result.stderr}")
            
            except subprocess.TimeoutExpired:
                print("⏰ (timeout)")
                failed_packages.append(package)
                self.setup_log.append(f"Timeout installing {package}")
            
            except Exception as e:
                print(f"❌ ({str(e)})")
                failed_packages.append(package)
                self.setup_log.append(f"Error installing {package}: {str(e)}")
        
        if failed_packages:
            print(f"\\n⚠️  Failed to install {len(failed_packages)} packages:")
            for package in failed_packages:
                print(f"   - {package}")
            
            retry = input("\\nRetry failed packages? (y/n): ").strip().lower()
            if retry == 'y':
                return self._install_packages(failed_packages)
        
        return len(failed_packages) == 0
    
    def _check_installation(self) -> bool:
        """Check if packages are properly installed."""
        print("\\n🔍 Checking installation...")
        
        missing_packages = []
        
        for package in self.requirements:
            try:
                importlib.import_module(package.replace('-', '_'))
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package}")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\\n⚠️  Missing packages: {', '.join(missing_packages)}")
            return False
        
        return True
    
    def _verify_installation(self) -> bool:
        """Verify that the installation works correctly."""
        print("\\n🧪 Verifying installation...")
        
        # Test basic imports
        test_imports = [
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("bitsandbytes", "BitsAndBytes"),
            ("accelerate", "Accelerate")
        ]
        
        for module, name in test_imports:
            try:
                importlib.import_module(module)
                print(f"✅ {name} import - OK")
            except ImportError as e:
                print(f"❌ {name} import failed: {e}")
                return False
        
        # Test CUDA if available
        if self.system_info["cuda_available"]:
            try:
                import torch
                if torch.cuda.is_available():
                    print("✅ CUDA functionality - OK")
                else:
                    print("⚠️  CUDA not available in PyTorch")
            except Exception as e:
                print(f"❌ CUDA test failed: {e}")
        
        # Test basic model loading
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print("✅ Model loading test - OK")
        except Exception as e:
            print(f"⚠️  Model loading test failed: {e}")
        
        return True
    
    def _configure_environment(self):
        """Configure environment variables and settings."""
        print("\\n⚙️  Configuring environment...")
        
        # Create configuration directory
        config_dir = Path.home() / ".llm_optimization"
        config_dir.mkdir(exist_ok=True)
        
        # Create configuration file
        config = {
            "setup_date": str(Path(__file__).stat().st_mtime),
            "system_info": self.system_info,
            "installed_packages": list(self.requirements.keys()),
            "cuda_available": self.system_info["cuda_available"]
        }
        
        with open(config_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Set environment variables
        env_vars = {
            "TOKENIZERS_PARALLELISM": "false",  # Avoid warnings
            "TRANSFORMERS_CACHE": str(config_dir / "cache"),
            "HF_HOME": str(config_dir / "huggingface")
        }
        
        for var, value in env_vars.items():
            os.environ[var] = value
            print(f"Set {var}={value}")
        
        print("✅ Environment configured")
    
    def _generate_setup_report(self):
        """Generate a setup report."""
        print("\\n📋 Generating setup report...")
        
        report_lines = [
            "# LLM Optimization Toolkit - Setup Report\\n",
            f"Setup completed on: {Path(__file__).stat().st_mtime}\\n\\n",
            "## System Information\\n"
        ]
        
        for key, value in self.system_info.items():
            report_lines.append(f"- {key}: {value}\\n")
        
        report_lines.append("\\n## Installation Log\\n")
        for log_entry in self.setup_log:
            report_lines.append(f"- {log_entry}\\n")
        
        report_lines.append("\\n## Next Steps\\n")
        report_lines.append("1. Try the quick start guide: `python -m llm_toolkit --help`\\n")
        report_lines.append("2. Run the interactive tutorial: `jupyter notebook tutorials/beginner/01_quantization_basics.ipynb`\\n")
        report_lines.append("3. Test with a simple quantization: `python -m llm_toolkit quantize --model gpt2 --method qlora`\\n")
        
        # Save report
        with open("setup_report.md", 'w') as f:
            f.writelines(report_lines)
        
        print("✅ Setup report saved to setup_report.md")
    
    def troubleshoot(self):
        """Provide troubleshooting assistance."""
        print("\\n🔧 Troubleshooting Assistant")
        print("=" * 40)
        
        # Common issues and solutions
        issues = {
            "CUDA out of memory": [
                "Reduce batch size",
                "Use gradient checkpointing",
                "Try smaller models",
                "Use CPU offloading"
            ],
            "Import errors": [
                "Check Python version (3.8+ required)",
                "Reinstall packages: pip install --upgrade --force-reinstall",
                "Check virtual environment activation",
                "Clear pip cache: pip cache purge"
            ],
            "Slow performance": [
                "Enable CUDA if available",
                "Use appropriate data types (float16)",
                "Increase batch size if memory allows",
                "Check for CPU bottlenecks"
            ],
            "Model loading errors": [
                "Check internet connection",
                "Clear Hugging Face cache",
                "Try different model names",
                "Check disk space"
            ]
        }
        
        print("Common issues and solutions:\\n")
        for issue, solutions in issues.items():
            print(f"**{issue}:**")
            for solution in solutions:
                print(f"  - {solution}")
            print()
        
        # System diagnostics
        print("\\n🔍 System Diagnostics:")
        print(f"Python: {sys.version}")
        print(f"Platform: {platform.platform()}")
        
        try:
            import torch
            print(f"PyTorch: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
        except ImportError:
            print("PyTorch: Not installed")
        
        try:
            import transformers
            print(f"Transformers: {transformers.__version__}")
        except ImportError:
            print("Transformers: Not installed")

def main():
    """Main function for the setup script."""
    parser = argparse.ArgumentParser(description="LLM Optimization Toolkit Environment Setup")
    parser.add_argument(
        "--mode",
        choices=["interactive", "auto", "check-only"],
        default="interactive",
        help="Setup mode"
    )
    parser.add_argument(
        "--troubleshoot",
        action="store_true",
        help="Run troubleshooting assistant"
    )
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup()
    
    if args.troubleshoot:
        setup.troubleshoot()
    else:
        success = setup.run_setup(args.mode)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()