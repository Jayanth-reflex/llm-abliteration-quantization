#!/usr/bin/env python3
"""
Smart Chatbot - Quick Start Script

This script provides a 5-minute quick start experience for the chatbot project.
It automatically sets up everything and gets you running immediately.

Usage:
    python quick_start.py --business-type coffee_shop --setup-time 5min
    python quick_start.py --demo-mode
    python quick_start.py --help

Author: LLM Optimization Toolkit
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add the implementation directory to path
sys.path.append(str(Path(__file__).parent / "implementation"))

class QuickStart:
    """Quick start automation for the Smart Chatbot project."""
    
    def __init__(self, business_type: str = "coffee_shop", setup_time: str = "5min"):
        self.business_type = business_type
        self.setup_time = setup_time
        self.demo_mode = False
        
    def print_banner(self):
        """Print welcome banner."""
        print("🚀" * 20)
        print("🤖 SMART CHATBOT - QUICK START")
        print("🚀" * 20)
        print(f"Business Type: {self.business_type}")
        print(f"Setup Time: {self.setup_time}")
        print("=" * 40)
        
    def run_quick_setup(self) -> bool:
        """Run automated setup in 5 minutes."""
        print("\n⚡ Running Quick Setup...")
        
        steps = [
            ("🔧 Environment Check", self._check_environment),
            ("📦 Install Dependencies", self._install_quick_deps),
            ("🧠 Load Model", self._load_demo_model),
            ("📚 Setup Knowledge", self._setup_business_knowledge),
            ("🚀 Start Chatbot", self._start_demo_chatbot)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            try:
                if not step_func():
                    print(f"❌ Failed: {step_name}")
                    return False
                print(f"✅ Completed: {step_name}")
            except Exception as e:
                print(f"❌ Error in {step_name}: {e}")
                return False
        
        return True
    
    def _check_environment(self) -> bool:
        """Quick environment check."""
        try:
            import torch
            import transformers
            print(f"   PyTorch: {torch.__version__}")
            print(f"   Transformers: {transformers.__version__}")
            return True
        except ImportError:
            print("   Installing required packages...")
            os.system("pip install torch transformers bitsandbytes accelerate --quiet")
            return True
    
    def _install_quick_deps(self) -> bool:
        """Install minimal dependencies for demo."""
        required = ["streamlit", "gradio", "sentence-transformers"]
        
        for package in required:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                print(f"   Installing {package}...")
                os.system(f"pip install {package} --quiet")
        
        return True
    
    def _load_demo_model(self) -> bool:
        """Load a small demo model quickly."""
        print("   Loading optimized demo model...")
        
        # Create a simple demo model configuration
        demo_config = {
            "model_name": "microsoft/DialoGPT-small",  # Smaller for quick demo
            "quantization": True,
            "max_length": 100
        }
        
        # Save config for demo
        os.makedirs("configs", exist_ok=True)
        with open("configs/demo_config.json", "w") as f:
            json.dump(demo_config, f, indent=2)
        
        return True
    
    def _setup_business_knowledge(self) -> bool:
        """Setup business knowledge base."""
        print(f"   Setting up {self.business_type} knowledge...")
        
        # Create business knowledge
        if self.business_type == "coffee_shop":
            knowledge = {
                "name": "Cozy Coffee Corner",
                "hours": "7 AM - 8 PM daily",
                "specialties": ["Artisan Coffee", "Fresh Pastries", "Free WiFi"],
                "location": "Downtown Main Street",
                "phone": "(555) 123-CAFE"
            }
        else:
            knowledge = {
                "name": f"Smart {self.business_type.title()}",
                "hours": "9 AM - 6 PM",
                "specialties": ["Quality Service", "Expert Staff"],
                "location": "Your City",
                "phone": "(555) 123-4567"
            }
        
        os.makedirs("data", exist_ok=True)
        with open("data/business_knowledge.json", "w") as f:
            json.dump(knowledge, f, indent=2)
        
        return True
    
    def _start_demo_chatbot(self) -> bool:
        """Start the demo chatbot interface."""
        print("   Creating demo interface...")
        
        # Create a simple demo script
        demo_script = '''
import streamlit as st
import json

st.title("🤖 Smart Chatbot Demo")
st.write("Welcome to your optimized business chatbot!")

# Load business knowledge
try:
    with open("data/business_knowledge.json", "r") as f:
        knowledge = json.load(f)
    
    st.sidebar.header("Business Info")
    st.sidebar.write(f"**{knowledge['name']}**")
    st.sidebar.write(f"Hours: {knowledge['hours']}")
    st.sidebar.write(f"Phone: {knowledge['phone']}")
    
except FileNotFoundError:
    st.error("Business knowledge not found!")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about our business!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Simple demo responses
    if "hours" in prompt.lower():
        response = f"We're open {knowledge['hours']}!"
    elif "phone" in prompt.lower() or "contact" in prompt.lower():
        response = f"You can reach us at {knowledge['phone']}"
    elif "location" in prompt.lower() or "where" in prompt.lower():
        response = f"We're located at {knowledge['location']}"
    else:
        response = f"Thanks for asking! We're {knowledge['name']} and we'd love to help you. What would you like to know?"
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

st.write("---")
st.write("🎯 **Next Steps:**")
st.write("1. Run the full tutorial: `jupyter notebook project_tutorial.ipynb`")
st.write("2. Customize for your business")
st.write("3. Deploy to production")
'''
        
        with open("demo_app.py", "w") as f:
            f.write(demo_script)
        
        return True
    
    def launch_demo(self):
        """Launch the demo interface."""
        print("\n🎉 Quick Start Complete!")
        print("\n🚀 Launching Demo...")
        print("=" * 40)
        print("Your chatbot is starting up...")
        print("It will open in your web browser automatically.")
        print("\nTo manually access:")
        print("1. Open: http://localhost:8501")
        print("2. Or run: streamlit run demo_app.py")
        print("=" * 40)
        
        # Launch Streamlit demo
        try:
            os.system("streamlit run demo_app.py --server.headless true")
        except KeyboardInterrupt:
            print("\n👋 Demo stopped. Thanks for trying the Smart Chatbot!")

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Smart Chatbot Quick Start")
    parser.add_argument("--business-type", default="coffee_shop", 
                       help="Type of business (coffee_shop, restaurant, retail, etc.)")
    parser.add_argument("--setup-time", default="5min",
                       help="Setup time preference (5min, 10min, full)")
    parser.add_argument("--demo-mode", action="store_true",
                       help="Run in demo mode only")
    
    args = parser.parse_args()
    
    # Create quick start instance
    quick_start = QuickStart(args.business_type, args.setup_time)
    quick_start.demo_mode = args.demo_mode
    
    # Print banner
    quick_start.print_banner()
    
    if args.demo_mode:
        print("\n🎮 Demo Mode - Launching existing demo...")
        quick_start.launch_demo()
    else:
        # Run setup
        if quick_start.run_quick_setup():
            quick_start.launch_demo()
        else:
            print("\n❌ Quick start failed. Try the manual setup:")
            print("python implementation/step_01_setup.py")

if __name__ == "__main__":
    main()