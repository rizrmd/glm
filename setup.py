#!/usr/bin/env python3
"""
Setup script for GLM-4.6 Runner
"""

from setuptools import setup, find_packages
import subprocess
import sys
import os

def install_system_deps():
    """Install system dependencies"""
    import platform
    
    system = platform.system()
    
    if system == 'Linux':
        print("Installing system dependencies for Linux...")
        try:
            if os.geteuid() == 0:
                subprocess.run(['apt-get', 'update'], check=True)
                subprocess.run(['apt-get', 'install', '-y', 
                              'pciutils', 'build-essential', 'cmake', 'curl', 
                              'libcurl4-openssl-dev', 'git', 'python3-pip'], check=True)
            else:
                print("Attempting with sudo...")
                subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                subprocess.run(['sudo', 'apt-get', 'install', '-y', 
                              'pciutils', 'build-essential', 'cmake', 'curl', 
                              'libcurl4-openssl-dev', 'git', 'python3-pip'], check=True)
        except subprocess.CalledProcessError:
            print("Failed to install system dependencies. Please install manually:")
            print("apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev git python3-pip")
    
    elif system == 'Darwin':
        print("Installing system dependencies for macOS...")
        try:
            result = subprocess.run(['which', 'brew'], capture_output=True)
            if result.returncode != 0:
                print("Installing Homebrew...")
                subprocess.run(['/bin/bash', '-c', 
                              '"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'], 
                             check=True)
            
            subprocess.run(['brew', 'install', 'cmake', 'curl', 'libgit2'], check=True)
        except subprocess.CalledProcessError:
            print("Failed to install system dependencies. Please install manually:")
            print("brew install cmake curl libgit2")

if __name__ == '__main__':
    # Install system dependencies first
    install_system_deps()
    
    # Install Python dependencies
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
    
    print("Setup completed successfully!")
    print("You can now run: python run_glm46_200k.py")