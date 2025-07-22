#!/usr/bin/env python3
"""
Smart Chatbot Project - Step 1: Environment Setup

This script sets up the complete environment for building an optimized chatbot.
It checks system requirements, installs dependencies, and verifies everything works.

Learning Objectives:
- Understand the requirements for LLM optimization
- Set up a reproducible development environment
- Verify GPU and quantization support
- Learn about system resource management

Author: LLM Optimization Toolkit
License: MIT
"""

import sys
import os
import subprocess
import platform
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Color codes for better terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKBLUE}ℹ️  {text}{Colors.ENDC}")

class EnvironmentSetup:
    """Handles complete environment setup for the chatbot project."""
    
    def __init__(self):
        self.system_info = {}
        self.requirements_met = True
        self.setup_log = []
        
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        print_info("Checking Python version...")
        
        version = sys.version_info
        required_major, required_minor = 3, 8
        
        if version.major >= required_major and version.minor >= required_minor:
            print_success(f"Python {version.major}.{version.minor}.{version.micro} - Compatible")
            self.system_info['python_version'] = f"{version.major}.{version.minor}.{version.micro}"
            return True
        else:
            print_error(f"Python {version.major}.{version.minor} is too old. Minimum required: {required_major}.{required_minor}")
            self.requirements_met = False
            return False
    
    def check_system_resources(self) -> bool:
        """Check system memory and storage."""
        print_info("Checking system resources...")
        
        try:
            import psutil
            
            # Check RAM
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            if memory_gb >= 8:
                print_success(f"System RAM: {memory_gb:.1f} GB - Sufficient")
                self.system_info['ram_gb'] = memory_gb
            else:
                print_warning(f"System RAM: {memory_gb:.1f} GB - Low (8GB+ recommended)")
                self.system_info['ram_gb'] = memory_gb
            
            # Check disk space
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024**3)
            
            if disk_free_gb >= 10:
                print_success(f"Free disk space: {disk_free_gb:.1f} GB - Sufficient")
                self.system_info['disk_free_gb'] = disk_free_gb
            else:
                print_warning(f"Free disk space: {disk_free_gb:.1f} GB - Low (10GB+ recommended)")
                self.system_info['disk_free_gb'] = disk_free_gb
            
            return True
            
        except ImportError:
            print_warning("psutil not available - installing...")
            self._install_package("psutil")
            return self.check_system_resources()
        except Exception as e:
            print_error(f"Error checking system resources: {e}")
            return False
    
    def check_gpu_support(self) -> bool:
        """Check for GPU availability and CUDA support."""
        print_info("Checking GPU support...")
        
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                print_success(f"GPU available: {gpu_name}")
                print_success(f"GPU memory: {gpu_memory:.1f} GB")
                print_success(f"CUDA version: {torch.version.cuda}")
                
                self.system_info['gpu_available'] = True
                self.system_info['gpu_name'] = gpu_name
                self.system_info['gpu_memory_gb'] = gpu_memory
                self.system_info['cuda_version'] = torch.version.cuda
                
                if gpu_memory >= 6:
                    print_success("GPU memory sufficient for quantized models")
                else:
                    print_warning("GPU memory low - consider using CPU or smaller models")
                
                return True
            else:
                print_warning("No GPU detected - will use CPU (slower but functional)")
                self.system_info['gpu_available'] = False
                return True
                
        except ImportError:
            print_warning("PyTorch not installed - will install during dependency setup")
            self.system_info['gpu_available'] = False
            return True
        except Exception as e:
            print_error(f"Error checking GPU: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install required Python packages."""
        print_info("Installing Python dependencies...")
        
        # Core requirements for the chatbot project
        requirements = [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "bitsandbytes>=0.41.0",
            "accelerate>=0.24.0",
            "peft>=0.6.0",
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.0",
            "streamlit>=1.28.0",
            "gradio>=3.50.0",
            "psutil>=5.9.0",
            "numpy>=1.21.0",
            "pandas>=1.5.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "tqdm>=4.64.0"
        ]
        
        print_info(f"Installing {len(requirements)} packages...")
        
        failed_packages = []
        
        for package in requirements:
            package_name = package.split('>=')[0]
            print(f"   Installing {package_name}...", end=" ")
            
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package, "--quiet"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    print(f"{Colors.OKGREEN}✓{Colors.ENDC}")
                else:
                    print(f"{Colors.FAIL}✗{Colors.ENDC}")
                    failed_packages.append(package_name)
                    
            except subprocess.TimeoutExpired:
                print(f"{Colors.WARNING}⏰{Colors.ENDC}")
                failed_packages.append(package_name)
            except Exception as e:
                print(f"{Colors.FAIL}✗{Colors.ENDC}")
                failed_packages.append(package_name)
        
        if failed_packages:
            print_warning(f"Failed to install: {', '.join(failed_packages)}")
            print_info("You can install them manually later with:")
            for pkg in failed_packages:
                print(f"   pip install {pkg}")
            return False
        else:
            print_success("All dependencies installed successfully!")
            return True
    
    def verify_installation(self) -> bool:
        """Verify that all required packages are working."""
        print_info("Verifying installation...")
        
        test_imports = [
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("bitsandbytes", "BitsAndBytes"),
            ("sentence_transformers", "Sentence Transformers"),
            ("faiss", "FAISS"),
            ("streamlit", "Streamlit"),
            ("gradio", "Gradio")
        ]
        
        failed_imports = []
        
        for module, name in test_imports:
            try:
                __import__(module)
                print_success(f"{name} import - OK")
            except ImportError as e:
                print_error(f"{name} import failed: {e}")
                failed_imports.append(name)
        
        if failed_imports:
            print_error(f"Failed imports: {', '.join(failed_imports)}")
            return False
        
        # Test quantization capability
        try:
            import torch
            from transformers import BitsAndBytesConfig
            
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print_success("Quantization configuration - OK")
            
        except Exception as e:
            print_error(f"Quantization test failed: {e}")
            return False
        
        print_success("All verifications passed!")
        return True
    
    def create_project_structure(self) -> bool:
        """Create the project directory structure."""
        print_info("Creating project structure...")
        
        directories = [
            "data",
            "models",
            "logs",
            "outputs",
            "checkpoints",
            "configs"
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(exist_ok=True)
                print_success(f"Created directory: {directory}")
            except Exception as e:
                print_error(f"Failed to create {directory}: {e}")
                return False
        
        # Create configuration file
        config = {
            "project_name": "smart_chatbot",
            "version": "1.0.0",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": self.system_info,
            "model_config": {
                "base_model": "microsoft/DialoGPT-medium",
                "quantization": {
                    "enabled": True,
                    "bits": 4,
                    "method": "qlora"
                }
            }
        }
        
        try:
            with open("configs/project_config.json", "w") as f:
                json.dump(config, f, indent=2)
            print_success("Created project configuration")
        except Exception as e:
            print_error(f"Failed to create config: {e}")
            return False
        
        return True
    
    def _install_package(self, package: str) -> bool:
        """Install a single package."""
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def generate_setup_report(self) -> str:
        """Generate a comprehensive setup report."""
        report = []
        report.append("# Smart Chatbot - Environment Setup Report")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("## System Information")
        for key, value in self.system_info.items():
            report.append(f"- {key}: {value}")
        report.append("")
        
        report.append("## Setup Status")
        if self.requirements_met:
            report.append("✅ All requirements met - Ready to proceed!")
        else:
            report.append("⚠️ Some requirements not met - Check warnings above")
        report.append("")
        
        report.append("## Next Steps")
        report.append("1. Run: python implementation/step_02_model_loading.py")
        report.append("2. Follow the step-by-step tutorial")
        report.append("3. Join our Discord for support: https://discord.gg/llm-optimization")
        
        return "\n".join(report)
    
    def run_complete_setup(self) -> bool:
        """Run the complete environment setup process."""
        print_header("🤖 Smart Chatbot - Environment Setup")
        
        print_info("Welcome to the Smart Chatbot project!")
        print_info("This setup will prepare your environment for LLM optimization.")
        print("")
        
        # Step 1: Check Python version
        if not self.check_python_version():
            return False
        
        # Step 2: Check system resources
        self.check_system_resources()
        
        # Step 3: Check GPU support
        self.check_gpu_support()
        
        # Step 4: Install dependencies
        if not self.install_dependencies():
            print_warning("Some packages failed to install, but continuing...")
        
        # Step 5: Verify installation
        if not self.verify_installation():
            print_error("Installation verification failed!")
            return False
        
        # Step 6: Create project structure
        if not self.create_project_structure():
            print_error("Failed to create project structure!")
            return False
        
        # Generate setup report
        report = self.generate_setup_report()
        
        try:
            with open("setup_report.md", "w") as f:
                f.write(report)
            print_success("Setup report saved to: setup_report.md")
        except Exception as e:
            print_warning(f"Could not save report: {e}")
        
        print_header("🎉 Setup Complete!")
        
        if self.requirements_met:
            print_success("Your environment is ready for the Smart Chatbot project!")
            print_info("Next steps:")
            print_info("1. Run: python implementation/step_02_model_loading.py")
            print_info("2. Or start the interactive tutorial: jupyter notebook project_tutorial.ipynb")
        else:
            print_warning("Setup completed with warnings. Check the issues above.")
        
        return self.requirements_met

def main():
    """Main function to run environment setup."""
    setup = EnvironmentSetup()
    success = setup.run_complete_setup()
    
    if success:
        print(f"\n{Colors.OKGREEN}🚀 Ready to build your optimized chatbot!{Colors.ENDC}")
        sys.exit(0)
    else:
        print(f"\n{Colors.FAIL}❌ Setup encountered issues. Please resolve them before continuing.{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()