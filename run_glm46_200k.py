#!/usr/bin/env python3
"""
GLM-4.6 Runner with Full 200K Context
Python version with automatic hardware detection and optimization
"""

import os
import sys
import subprocess
import shutil
import argparse
import platform
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

# Auto-install required dependencies
def auto_install_dependencies():
    """Automatically install required Python dependencies if not already installed"""
    required_packages = ['psutil']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"{Colors.GREEN}[OK]{Colors.NC} {package} already installed")
        except ImportError:
            print(f"{Colors.YELLOW}[INFO]{Colors.NC} Installing {package}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', package])
                print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"{Colors.RED}[ERROR]{Colors.NC} Failed to install {package}: {e}")
                sys.exit(1)

# Install dependencies before importing psutil
auto_install_dependencies()
import psutil

class HardwareDetector:
    """Detect and analyze system hardware capabilities"""
    
    def __init__(self):
        self.system_info = self._detect_system()
        self.gpu_info = self._detect_gpu()
        self.memory_info = self._detect_memory()
        self.cpu_info = self._detect_cpu()
        self.storage_info = self._detect_storage()
    
    def _detect_system(self) -> Dict:
        """Detect basic system information"""
        return {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
    
    def _detect_gpu(self) -> Dict:
        """Detect GPU information"""
        gpu_info = {
            'nvidia_available': False,
            'amd_available': False,
            'apple_silicon': False,
            'gpu_memory': 0,
            'gpu_count': 0,
            'gpu_names': []
        }
        
        # Check for NVIDIA GPU
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_info['nvidia_available'] = True
                lines = result.stdout.strip().split('\n')
                gpu_info['gpu_count'] = len(lines)
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        gpu_info['gpu_names'].append(parts[0])
                        gpu_info['gpu_memory'] += int(parts[1])
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check for Apple Silicon
        if self.system_info['platform'] == 'Darwin':
            try:
                result = subprocess.run(['sysctl', '-n', 'hw.optional.gpu'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip() != '0':
                    gpu_info['apple_silicon'] = True
                    gpu_info['gpu_count'] = 1
                    # Apple Silicon GPU memory is shared with system memory
                    gpu_info['gpu_memory'] = self.memory_info['total_gb'] * 1024
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        # Check for AMD GPU on Linux
        if self.system_info['platform'] == 'Linux':
            try:
                result = subprocess.run(['lspci'], capture_output=True, text=True)
                if 'AMD' in result.stdout and 'Radeon' in result.stdout:
                    gpu_info['amd_available'] = True
                    gpu_info['gpu_count'] = result.stdout.count('AMD')
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
        
        return gpu_info
    
    def _detect_memory(self) -> Dict:
        """Detect memory information"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'percent_used': memory.percent
        }
    
    def _detect_cpu(self) -> Dict:
        """Detect CPU information"""
        cpu_count = psutil.cpu_count(logical=True)
        physical_count = psutil.cpu_count(logical=False)
        
        cpu_info = {
            'logical_cores': cpu_count,
            'physical_cores': physical_count,
            'max_frequency': 0,
            'current_frequency': 0,
            'cpu_usage': psutil.cpu_percent(interval=1)
        }
        
        # Get frequency information
        try:
            if self.system_info['platform'] == 'Linux':
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'cpu MHz' in line:
                            cpu_info['max_frequency'] = float(line.split(':')[1].strip())
                            break
            elif self.system_info['platform'] == 'Darwin':
                result = subprocess.run(['sysctl', '-n', 'hw.cpufrequency'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    cpu_info['max_frequency'] = float(result.stdout.strip()) / 1000000
        except (subprocess.TimeoutExpired, FileNotFoundError, IOError):
            pass
        
        return cpu_info
    
    def _detect_storage(self) -> Dict:
        """Detect storage information"""
        disk_usage = psutil.disk_usage('.')
        return {
            'total_gb': round(disk_usage.total / (1024**3), 2),
            'free_gb': round(disk_usage.free / (1024**3), 2),
            'used_gb': round(disk_usage.used / (1024**3), 2),
            'percent_used': round((disk_usage.used / disk_usage.total) * 100, 2)
        }
    
    def get_optimal_settings(self) -> Dict:
        """Generate optimal settings based on detected hardware"""
        settings = {
            'threads': self.cpu_info['logical_cores'],
            'gpu_layers': 0,
            'use_flash_attention': False,
            'cache_quantization': False,
            'batch_size': 512,
            'context_size': 200000,
            'recommended_quant': 'UD-Q2_K_XL'
        }
        
        # GPU optimization
        if self.gpu_info['nvidia_available']:
            settings['gpu_layers'] = 999
            settings['use_flash_attention'] = True
            if self.gpu_info['gpu_memory'] >= 8000:  # 8GB+
                settings['cache_quantization'] = True
                settings['batch_size'] = 1024
            if self.gpu_info['gpu_memory'] >= 16000:  # 16GB+
                settings['recommended_quant'] = 'UD-Q4_K_XL'
        
        elif self.gpu_info['apple_silicon']:
            settings['gpu_layers'] = 999
            settings['use_flash_attention'] = True
            settings['cache_quantization'] = True
            if self.memory_info['total_gb'] >= 16:
                settings['recommended_quant'] = 'UD-Q4_K_XL'
        
        # Memory optimization
        if self.memory_info['total_gb'] >= 64:
            settings['batch_size'] = 2048
        elif self.memory_info['total_gb'] >= 32:
            settings['batch_size'] = 1024
        elif self.memory_info['total_gb'] < 16:
            settings['recommended_quant'] = 'UD-TQ1_0'
            settings['context_size'] = min(100000, settings['context_size'])
        
        # CPU optimization
        if self.cpu_info['physical_cores'] >= 16:
            settings['threads'] = self.cpu_info['physical_cores'] - 2
        elif self.cpu_info['physical_cores'] >= 8:
            settings['threads'] = self.cpu_info['physical_cores'] - 1
        
        return settings

class GLMRunner:
    """Main GLM-4.6 runner class"""
    
    def __init__(self):
        self.hardware = HardwareDetector()
        self.config = {
            'model_repo': 'unsloth/GLM-4.6-GGUF',
            'model_dir': 'unsloth/GLM-4.6-GGUF',
            'llama_cpp_dir': 'llama.cpp',
            'quant_type': 'UD-Q2_K_XL',
            'context_size': 200000,
            'temperature': 1.0,
            'top_p': 0.95,
            'top_k': 40,
            'seed': 3407
        }
        self.optimal_settings = self.hardware.get_optimal_settings()
        
    def print_status(self, message: str, color: str = Colors.BLUE):
        """Print colored status message"""
        print(f"{color}[INFO]{Colors.NC} {message}")
    
    def print_success(self, message: str):
        """Print success message"""
        print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")
    
    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")
    
    def print_error(self, message: str):
        """Print error message"""
        print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")
    
    def check_requirements(self) -> bool:
        """Check system requirements"""
        self.print_status("Checking system requirements...")
        
        # Check disk space
        if self.hardware.storage_info['free_gb'] < 150:
            self.print_error(f"Insufficient disk space. Need at least 150GB, available: {self.hardware.storage_info['free_gb']}GB")
            return False
        
        # Check memory
        if self.hardware.memory_info['total_gb'] < 32:
            self.print_warning(f"Low RAM detected ({self.hardware.memory_info['total_gb']}GB). For optimal 200K context performance, 32GB+ RAM is recommended")
        
        # Display hardware info
        self.print_status(f"System: {self.hardware.system_info['platform']} {self.hardware.system_info['architecture']}")
        self.print_status(f"CPU: {self.hardware.cpu_info['physical_cores']} cores ({self.hardware.cpu_info['logical_cores']} logical)")
        self.print_status(f"Memory: {self.hardware.memory_info['total_gb']}GB total, {self.hardware.memory_info['available_gb']}GB available")
        self.print_status(f"Storage: {self.hardware.storage_info['free_gb']}GB free")
        
        if self.hardware.gpu_info['nvidia_available']:
            self.print_status(f"GPU: {self.hardware.gpu_info['gpu_count']} NVIDIA GPU(s) with {self.hardware.gpu_info['gpu_memory']}MB total memory")
        elif self.hardware.gpu_info['apple_silicon']:
            self.print_status("GPU: Apple Silicon GPU detected")
        elif self.hardware.gpu_info['amd_available']:
            self.print_status(f"GPU: {self.hardware.gpu_info['gpu_count']} AMD GPU(s) detected")
        else:
            self.print_warning("No GPU detected. Will run in CPU-only mode")
        
        self.print_success("System requirements check completed")
        return True
    
    def check_command_exists(self, command: str) -> bool:
        """Check if a command exists on the system"""
        try:
            result = subprocess.run(['which', command], capture_output=True, text=True)
            return result.returncode == 0 and len(result.stdout.strip()) > 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def check_deb_package_installed(self, package: str) -> bool:
        """Check if a Debian package is installed"""
        try:
            result = subprocess.run(['dpkg', '-l', package], capture_output=True, text=True)
            return result.returncode == 0 and 'ii' in result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def check_brew_package_installed(self, package: str) -> bool:
        """Check if a Homebrew package is installed"""
        try:
            result = subprocess.run(['brew', 'list', package], capture_output=True)
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def check_python_package_installed(self, package: str) -> bool:
        """Check if a Python package is installed"""
        try:
            __import__(package)
            return True
        except ImportError:
            return False
    
    def install_dependencies(self):
        """Install system dependencies only if not already installed"""
        self.print_status("Checking and installing dependencies...")
        
        system = self.hardware.system_info['platform']
        
        if system == 'Linux':
            packages = ['pciutils', 'build-essential', 'cmake', 'curl', 
                       'libcurl4-openssl-dev', 'git', 'python3-pip']
            packages_to_install = []
            
            for package in packages:
                if not self.check_deb_package_installed(package):
                    packages_to_install.append(package)
                else:
                    self.print_status(f"✓ {package} already installed")
            
            if packages_to_install:
                self.print_status(f"Installing missing packages: {', '.join(packages_to_install)}")
                try:
                    if os.geteuid() == 0:
                        subprocess.run(['apt-get', 'update'], check=True)
                        subprocess.run(['apt-get', 'install', '-y'] + packages_to_install, check=True)
                    else:
                        self.print_warning("Not running as root. Attempting with sudo...")
                        subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                        subprocess.run(['sudo', 'apt-get', 'install', '-y'] + packages_to_install, check=True)
                    self.print_success("System packages installed successfully")
                except subprocess.CalledProcessError as e:
                    self.print_error(f"Failed to install dependencies: {e}")
                    self.print_status("Please install manually: apt-get install " + " ".join(packages_to_install))
                    sys.exit(1)
            else:
                self.print_success("All system packages already installed")
        
        elif system == 'Darwin':
            # Check if Homebrew is installed
            if not self.check_command_exists('brew'):
                self.print_status("Installing Homebrew...")
                try:
                    subprocess.run(['/bin/bash', '-c', 
                                  '"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'], 
                                 check=True)
                    self.print_success("Homebrew installed successfully")
                except subprocess.CalledProcessError as e:
                    self.print_error(f"Failed to install Homebrew: {e}")
                    sys.exit(1)
            else:
                self.print_status("✓ Homebrew already installed")
            
            # Check and install Homebrew packages
            brew_packages = ['cmake', 'curl', 'libgit2']
            packages_to_install = []
            
            for package in brew_packages:
                if not self.check_brew_package_installed(package):
                    packages_to_install.append(package)
                else:
                    self.print_status(f"✓ {package} already installed")
            
            if packages_to_install:
                self.print_status(f"Installing missing Homebrew packages: {', '.join(packages_to_install)}")
                try:
                    subprocess.run(['brew', 'install'] + packages_to_install, check=True)
                    self.print_success("Homebrew packages installed successfully")
                except subprocess.CalledProcessError as e:
                    self.print_error(f"Failed to install Homebrew packages: {e}")
                    self.print_status("Please install manually: brew install " + " ".join(packages_to_install))
                    sys.exit(1)
            else:
                self.print_success("All Homebrew packages already installed")
        
        # Install Python dependencies
        python_packages = ['huggingface_hub', 'hf_transfer']
        packages_to_install = []
        
        for package in python_packages:
            if not self.check_python_package_installed(package):
                packages_to_install.append(package)
            else:
                self.print_status(f"✓ {package} already installed")
        
        if packages_to_install:
            self.print_status(f"Installing missing Python packages: {', '.join(packages_to_install)}")
            for package in packages_to_install:
                try:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', '--quiet', package], check=True)
                    self.print_status(f"✓ {package} installed successfully")
                except subprocess.CalledProcessError as e:
                    self.print_error(f"Failed to install {package}: {e}")
                    sys.exit(1)
            self.print_success("Python packages installed successfully")
        else:
            self.print_success("All Python packages already installed")
        
        self.print_success("Dependency check completed")
    
    def check_llama_cpp_built(self) -> bool:
        """Check if llama.cpp is already built"""
        required_binaries = ['llama-cli', 'llama-server']
        for binary in required_binaries:
            binary_path = Path(self.config['llama_cpp_dir']) / binary
            if not binary_path.exists():
                return False
        return True
    
    def build_llama_cpp(self):
        """Build llama.cpp with optimal settings"""
        if self.check_llama_cpp_built():
            self.print_success("llama.cpp already built")
            return
            
        self.print_status("Building llama.cpp with optimized settings...")
        
        if not os.path.exists(self.config['llama_cpp_dir']):
            self.print_status("Cloning llama.cpp repository...")
            subprocess.run(['git', 'clone', 'https://github.com/ggerganov/llama.cpp'], check=True)
        else:
            self.print_status("llama.cpp repository already exists, updating...")
            os.chdir(self.config['llama_cpp_dir'])
            subprocess.run(['git', 'pull'], check=True)
            os.chdir('..')
        
        os.chdir(self.config['llama_cpp_dir'])
        
        # Configure CMake based on hardware
        cmake_args = [
            'cmake', '-B', 'build',
            '-DBUILD_SHARED_LIBS=OFF',
            '-DLLAMA_CURL=ON',
            f'-DLLAMA_MAX_CONTEXT={self.config["context_size"]}'
        ]
        
        if self.hardware.gpu_info['nvidia_available']:
            cmake_args.extend(['-DGGML_CUDA=ON', '-DGGML_CUDA_FA_ALL_QUANTS=ON'])
        else:
            cmake_args.append('-DGGML_CUDA=OFF')
        
        subprocess.run(cmake_args, check=True)
        
        # Build with optimal parallelization
        build_jobs = min(self.hardware.cpu_info['logical_cores'], 16)
        subprocess.run([
            'cmake', '--build', 'build', '--config', 'Release', 
            f'-j{build_jobs}', '--clean-first',
            '--target', 'llama-quantize', 'llama-cli', 'llama-gguf-split', 
            'llama-mtmd-cli', 'llama-server'
        ], check=True)
        
        # Copy binaries
        for binary in ['llama-quantize', 'llama-cli', 'llama-gguf-split', 
                      'llama-mtmd-cli', 'llama-server']:
            src = f'build/bin/{binary}'
            dst = binary
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        os.chdir('..')
        self.print_success("llama.cpp built successfully with optimized settings")
    
    def get_optimized_model_files(self) -> List[str]:
        """Get list of optimized model files for the current quantization"""
        # Map quantization types to their specific file patterns
        quant_file_map = {
            'UD-TQ1_0': ['*UD-TQ1_0*.gguf'],
            'UD-IQ1_S': ['*UD-IQ1_S*.gguf'],
            'UD-IQ1_M': ['*UD-IQ1_M*.gguf'],
            'UD-IQ2_XXS': ['*UD-IQ2_XXS*.gguf'],
            'UD-Q2_K_XL': ['*UD-Q2_K_XL*.gguf'],
            'UD-IQ3_XXS': ['*UD-IQ3_XXS*.gguf'],
            'UD-Q3_K_XL': ['*UD-Q3_K_XL*.gguf'],
            'UD-Q4_K_XL': ['*UD-Q4_K_XL*.gguf'],
            'UD-Q5_K_XL': ['*UD-Q5_K_XL*.gguf']
        }
        
        return quant_file_map.get(self.config['quant_type'], [f"*{self.config['quant_type']}*.gguf"])
    
    def download_model(self, parallel=False):
        """Download only optimized GLM-4.6 model files if not already present"""
        model_dir = Path(self.config['model_dir'])
        model_patterns = self.get_optimized_model_files()
        
        # Check if model files already exist
        existing_files = []
        for pattern in model_patterns:
            existing_files.extend(model_dir.glob(pattern))
        
        if existing_files:
            total_size = sum(f.stat().st_size for f in existing_files)
            size_gb = total_size / (1024**3)
            self.print_success(f"Optimized model files already exist ({size_gb:.1f}GB total)")
            return True
        
        if not parallel:
            self.print_status(f"Downloading optimized GLM-4.6 model ({self.config['quant_type']} quantization)...")
        
        # Create download script with optimized file patterns
        patterns_str = ', '.join([f'"{p}"' for p in model_patterns])
        download_script = f'''
import os
import sys
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

repo_id = "{self.config['model_repo']}"
local_dir = "{self.config['model_dir']}"
quant_type = "{self.config['quant_type']}"
patterns = [{patterns_str}]

if not {parallel}:
    print(f"Downloading optimized {{quant_type}} model files...")
else:
    print("[PARALLEL] Starting optimized model download in background...")

try:
    # Try to download specific files first for more efficiency
    downloaded_files = []
    for pattern in patterns:
        try:
            # List files matching the pattern
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem()
            repo_files = fs.glob(f"{{repo_id}}/{{pattern}}")
            
            for file_path in repo_files:
                if file_path.endswith('.gguf'):
                    local_path = Path(local_dir) / Path(file_path).name
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if not {parallel}:
                        print(f"Downloading {{Path(file_path).name}}...")
                    
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=file_path,
                        local_dir=local_dir,
                        resume_download=True
                    )
                    downloaded_files.append(file_path)
        
        except Exception as e:
            # Fallback to snapshot_download if specific download fails
            if not {parallel}:
                print(f"Specific download failed, trying fallback: {{e}}")
            else:
                print(f"[PARALLEL] Specific download failed, trying fallback: {{e}}")
            
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                allow_patterns=patterns,
                resume_download=True
            )
    
    if not {parallel}:
        print("Optimized model download completed successfully!")
    else:
        print("[PARALLEL] Optimized model download completed successfully!")
        
except Exception as e:
    if not {parallel}:
        print(f"Download failed: {{e}}")
    else:
        print(f"[PARALLEL] Model download failed: {{e}}")
    sys.exit(1)
'''
        
        with open('download_model.py', 'w') as f:
            f.write(download_script)
        
        if parallel:
            # Run in background thread
            def download_worker():
                try:
                    subprocess.run([sys.executable, 'download_model.py'], check=True)
                except subprocess.CalledProcessError:
                    pass
                finally:
                    if os.path.exists('download_model.py'):
                        os.remove('download_model.py')
            
            thread = threading.Thread(target=download_worker, daemon=True)
            thread.start()
            return thread
        else:
            subprocess.run([sys.executable, 'download_model.py'], check=True)
            os.remove('download_model.py')
            self.print_success("Optimized model download completed")
            return True
    
    def merge_model_files(self) -> str:
        """Merge split GGUF files if necessary"""
        self.print_status("Checking if optimized model files need merging...")
        
        model_dir = Path(self.config['model_dir'])
        model_patterns = self.get_optimized_model_files()
        
        # Check for split files across all patterns
        split_files = []
        for pattern in model_patterns:
            split_files.extend(model_dir.glob(f"{pattern.replace('*', '')}-00001-of-*.gguf"))
        
        if split_files:
            self.print_status("Merging split GGUF files...")
            first_file = split_files[0]
            merged_file = model_dir / f"GLM-4.6-{self.config['quant_type']}-merged.gguf"
            
            subprocess.run([
                f'./{self.config["llama_cpp_dir"]}/llama-gguf-split',
                '--merge', str(first_file), str(merged_file)
            ], check=True)
            
            model_path = str(merged_file)
            self.print_success("Optimized model files merged successfully")
        else:
            # Find the main optimized model file
            model_files = []
            for pattern in model_patterns:
                model_files.extend(model_dir.glob(pattern))
            
            # Filter out split files
            model_files = [f for f in model_files if '-0000' not in f.name]
            
            if model_files:
                # Choose the largest file (usually the main model)
                model_files.sort(key=lambda x: x.stat().st_size, reverse=True)
                model_path = str(model_files[0])
            else:
                # Fallback to any matching file
                for pattern in model_patterns:
                    fallback_files = list(model_dir.glob(pattern))
                    if fallback_files:
                        model_path = str(fallback_files[0])
                        break
                else:
                    model_path = ""
            
            if model_path:
                self.print_success(f"Using optimized model file: {Path(model_path).name}")
            else:
                self.print_error("No model files found!")
                sys.exit(1)
        
        return model_path
    
    def build_command(self, model_path: str, server_mode: bool = False) -> List[str]:
        """Build optimized command based on hardware"""
        if server_mode:
            cmd = [
                f'./{self.config["llama_cpp_dir"]}/llama-server',
                '--model', model_path,
                '--alias', 'GLM-4.6-200K',
                '--ctx-size', str(self.config['context_size']),
                '--temp', str(self.config['temperature']),
                '--top-p', str(self.config['top_p']),
                '--top-k', str(self.config['top_k']),
                '--port', '8001',
                '--jinja',
                '--host', '0.0.0.0'
            ]
        else:
            cmd = [
                f'./{self.config["llama_cpp_dir"]}/llama-cli',
                '--model', model_path,
                '--jinja',
                '--ctx-size', str(self.config['context_size']),
                '--temp', str(self.config['temperature']),
                '--top-p', str(self.config['top_p']),
                '--top-k', str(self.config['top_k']),
                '--seed', str(self.config['seed']),
                '--in-prefix', ' ',
                '--color', '-i'
            ]
        
        # Add hardware-specific optimizations
        if self.hardware.gpu_info['nvidia_available']:
            cmd.extend(['--n-gpu-layers', str(self.optimal_settings['gpu_layers'])])
            cmd.extend(['-ot', '.ffn_.*_exps.=CPU'])
            
            if self.optimal_settings['cache_quantization']:
                cmd.extend(['--cache-type-k', 'q4_1', '--cache-type-v', 'q4_1'])
            
            if self.optimal_settings['use_flash_attention']:
                cmd.append('--flash-attn')
        
        elif self.hardware.gpu_info['apple_silicon']:
            cmd.extend(['--n-gpu-layers', str(self.optimal_settings['gpu_layers'])])
            if self.optimal_settings['use_flash_attention']:
                cmd.append('--flash-attn')
        
        else:
            # CPU-only optimization
            cmd.extend(['--threads', str(self.optimal_settings['threads'])])
        
        return cmd
    
    def run_model(self, model_path: str):
        """Run GLM-4.6 in interactive mode"""
        self.print_status("Starting GLM-4.6 with 200K context...")
        
        os.environ['LLAMA_CACHE'] = self.config['model_dir']
        
        cmd = self.build_command(model_path, server_mode=False)
        
        self.print_status("Running with command:")
        print(f"{Colors.YELLOW}{' '.join(cmd)}{Colors.NC}")
        self.print_status("Note: First load may take several minutes for 200K context initialization...")
        
        subprocess.run(cmd)
    
    def run_server(self, model_path: str):
        """Run GLM-4.6 in server mode"""
        self.print_status("Starting GLM-4.6 server with 200K context...")
        
        os.environ['LLAMA_CACHE'] = self.config['model_dir']
        
        cmd = self.build_command(model_path, server_mode=True)
        
        self.print_status("Starting server with command:")
        print(f"{Colors.YELLOW}{' '.join(cmd)}{Colors.NC}")
        
        subprocess.run(cmd)
    
    def show_usage(self):
        """Show usage information"""
        print("Usage: python run_glm46_200k.py [OPTIONS]")
        print("")
        print("Options:")
        print("  -h, --help          Show this help message")
        print("  -s, --server        Run in server mode (default: interactive mode)")
        print("  -q, --quant TYPE    Quantization type (default: UD-Q2_K_XL)")
        print("  --skip-deps         Skip dependency installation")
        print("  --skip-build        Skip llama.cpp build")
        print("  --skip-download     Skip model download")
        print("  --hardware-info     Show detailed hardware information")
        print("")
        print("Available quantization types:")
        print("  UD-TQ1_0    (1.66bit, ~84GB) - Smallest size")
        print("  UD-IQ1_S    (1.78bit, ~96GB)")
        print("  UD-IQ1_M    (1.93bit, ~107GB)")
        print("  UD-IQ2_XXS  (2.42bit, ~115GB)")
        print("  UD-Q2_K_XL  (2.71bit, ~135GB) - Recommended balance")
        print("  UD-IQ3_XXS  (3.12bit, ~145GB)")
        print("  UD-Q3_K_XL  (3.5bit, ~158GB)")
        print("  UD-Q4_K_XL  (4.5bit, ~204GB)")
        print("  UD-Q5_K_XL  (5.5bit, ~252GB)")
        print("")
        print("Hardware-optimized recommendations:")
        print(f"  Recommended quantization: {self.optimal_settings['recommended_quant']}")
        print(f"  Optimal threads: {self.optimal_settings['threads']}")
        print(f"  GPU layers: {self.optimal_settings['gpu_layers']}")
        print(f"  Flash attention: {self.optimal_settings['use_flash_attention']}")
        print(f"  Cache quantization: {self.optimal_settings['cache_quantization']}")
        print("")
        print("Example:")
        print("  python run_glm46_200k.py                    # Run interactive mode")
        print("  python run_glm46_200k.py --server          # Run server mode")
        print("  python run_glm46_200k.py -q UD-Q4_K_XL     # Use 4.5bit quantization")
    
    def show_hardware_info(self):
        """Show detailed hardware information"""
        print("=== HARDWARE INFORMATION ===")
        print(f"Platform: {self.hardware.system_info['platform']} {self.hardware.system_info['architecture']}")
        print(f"Python: {self.hardware.system_info['python_version']}")
        print("")
        print("CPU:")
        print(f"  Physical cores: {self.hardware.cpu_info['physical_cores']}")
        print(f"  Logical cores: {self.hardware.cpu_info['logical_cores']}")
        print(f"  Max frequency: {self.hardware.cpu_info['max_frequency']:.2f} MHz")
        print(f"  Current usage: {self.hardware.cpu_info['cpu_usage']:.1f}%")
        print("")
        print("Memory:")
        print(f"  Total: {self.hardware.memory_info['total_gb']} GB")
        print(f"  Available: {self.hardware.memory_info['available_gb']} GB")
        print(f"  Used: {self.hardware.memory_info['used_gb']} GB ({self.hardware.memory_info['percent_used']:.1f}%)")
        print("")
        print("Storage:")
        print(f"  Total: {self.hardware.storage_info['total_gb']} GB")
        print(f"  Free: {self.hardware.storage_info['free_gb']} GB")
        print(f"  Used: {self.hardware.storage_info['used_gb']} GB ({self.hardware.storage_info['percent_used']:.1f}%)")
        print("")
        print("GPU:")
        if self.hardware.gpu_info['nvidia_available']:
            print(f"  Type: NVIDIA")
            print(f"  Count: {self.hardware.gpu_info['gpu_count']}")
            print(f"  Total memory: {self.hardware.gpu_info['gpu_memory']} MB")
            print(f"  Models: {', '.join(self.hardware.gpu_info['gpu_names'])}")
        elif self.hardware.gpu_info['apple_silicon']:
            print(f"  Type: Apple Silicon")
            print(f"  Memory: Shared with system memory")
        elif self.hardware.gpu_info['amd_available']:
            print(f"  Type: AMD")
            print(f"  Count: {self.hardware.gpu_info['gpu_count']}")
        else:
            print("  No GPU detected")
        print("")
        print("=== OPTIMAL SETTINGS ===")
        for key, value in self.optimal_settings.items():
            print(f"  {key}: {value}")
    
    def wait_for_download(self, download_thread):
        """Wait for parallel download to complete with progress indication"""
        if download_thread is None:
            return True
            
        self.print_status("Waiting for model download to complete...")
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        idx = 0
        
        while download_thread.is_alive():
            print(f"\r{Colors.YELLOW}[DOWNLOAD]{Colors.NC} {spinner[idx % len(spinner)]} Downloading model... ", end='', flush=True)
            idx += 1
            time.sleep(0.1)
            download_thread.join(timeout=0.1)
        
        print(f"\r{Colors.GREEN}[SUCCESS]{Colors.NC} Model download completed!{' ' * 20}")
        return True
    
    def main(self):
        """Main execution function"""
        parser = argparse.ArgumentParser(description='GLM-4.6 Runner with 200K Context')
        parser.add_argument('-s', '--server', action='store_true', help='Run in server mode')
        parser.add_argument('-q', '--quant', type=str, default=self.optimal_settings['recommended_quant'],
                          help='Quantization type')
        parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
        parser.add_argument('--skip-build', action='store_true', help='Skip llama.cpp build')
        parser.add_argument('--skip-download', action='store_true', help='Skip model download')
        parser.add_argument('--hardware-info', action='store_true', help='Show detailed hardware information')
        parser.add_argument('--no-parallel', action='store_true', help='Disable parallel model download')
        parser.add_argument('--debug', action='store_true', help='Enable debug mode to see what\'s being checked')
        
        args = parser.parse_args()
        
        if args.hardware_info:
            self.show_hardware_info()
            return
        
        self.config['quant_type'] = args.quant
        
        self.print_status("GLM-4.6 Runner with 200K Context")
        self.print_status(f"Quantization: {self.config['quant_type']}")
        self.print_status(f"Context Size: {self.config['context_size']}")
        self.print_status(f"Server Mode: {args.server}")
        print("")
        
        # Execute steps
        if not self.check_requirements():
            sys.exit(1)
        
        # Start parallel download if enabled and not skipped
        download_thread = None
        if not args.skip_download and not args.no_parallel:
            self.print_status("Starting parallel model download...")
            download_thread = self.download_model(parallel=True)
        
        # Install dependencies while model downloads
        if not args.skip_deps:
            self.install_dependencies()
        
        # Build llama.cpp while model downloads
        if not args.skip_build:
            self.build_llama_cpp()
        
        # Wait for download to complete if it was running in parallel
        if download_thread:
            self.wait_for_download(download_thread)
        elif not args.skip_download:
            self.download_model(parallel=False)
        
        model_path = self.merge_model_files()
        
        if args.server:
            self.run_server(model_path)
        else:
            self.run_model(model_path)

if __name__ == '__main__':
    runner = GLMRunner()
    runner.main()