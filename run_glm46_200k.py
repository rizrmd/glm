#!/usr/bin/env python3
"""
GLM-4.6 Runner with Full 200K Context (Linux Only)
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
import requests
import tempfile
import zipfile
import tarfile
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
                if result.returncode == 0 and result.stdout.strip():
                    try:
                        cpu_info['max_frequency'] = float(result.stdout.strip()) / 1000000
                    except ValueError:
                        pass
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
        
        # GPU optimization - enhanced for high-end GPUs
        if self.gpu_info['nvidia_available']:
            settings['gpu_layers'] = 999
            settings['use_flash_attention'] = True
            
            # High-end GPU optimization (H100, H200, A100, etc.)
            if self.gpu_info['gpu_memory'] >= 80000:  # 80GB+ (H100/H200)
                settings['cache_quantization'] = True
                settings['batch_size'] = 1024  # Further optimized for H200 latency
                settings['recommended_quant'] = 'UD-Q4_K_XL'  # Better performance/quality balance
                settings['use_tensor_cores'] = True
            elif self.gpu_info['gpu_memory'] >= 40000:  # 40GB+ (A100)
                settings['cache_quantization'] = True
                settings['batch_size'] = 3072
                settings['recommended_quant'] = 'UD-Q4_K_XL'
                settings['use_tensor_cores'] = True
            elif self.gpu_info['gpu_memory'] >= 16000:  # 16GB+
                settings['cache_quantization'] = True
                settings['batch_size'] = 2048
                settings['recommended_quant'] = 'UD-Q4_K_XL'
            elif self.gpu_info['gpu_memory'] >= 8000:  # 8GB+
                settings['cache_quantization'] = True
                settings['batch_size'] = 1024
        
        elif self.gpu_info['apple_silicon']:
            settings['gpu_layers'] = 999
            settings['use_flash_attention'] = True
            settings['cache_quantization'] = True
            if self.memory_info['total_gb'] >= 16:
                settings['recommended_quant'] = 'UD-Q4_K_XL'
        
        # Memory optimization - more aggressive for high-memory systems
        if self.memory_info['total_gb'] >= 256:
            settings['batch_size'] = min(settings['batch_size'], 8192)
        elif self.memory_info['total_gb'] >= 128:
            settings['batch_size'] = min(settings['batch_size'], 6144)
        elif self.memory_info['total_gb'] >= 64:
            settings['batch_size'] = min(settings['batch_size'], 4096)
        elif self.memory_info['total_gb'] >= 32:
            settings['batch_size'] = min(settings['batch_size'], 2048)
        elif self.memory_info['total_gb'] < 16:
            settings['recommended_quant'] = 'UD-TQ1_0'
            settings['context_size'] = min(100000, settings['context_size'])
        
        # CPU optimization
        if self.cpu_info['physical_cores'] >= 32:
            settings['threads'] = self.cpu_info['physical_cores'] - 4
        elif self.cpu_info['physical_cores'] >= 16:
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
        
        # Check disk space (temporarily bypass for testing)
        if self.hardware.storage_info['free_gb'] < 10:
            self.print_error(f"Insufficient disk space. Need at least 10GB, available: {self.hardware.storage_info['free_gb']}GB")
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
            self.print_error("No GPU detected. This script requires NVIDIA or AMD GPU support.")
            self.print_status("CPU-only mode is not supported.")
            sys.exit(1)
        
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
            self.print_error("macOS is not supported. This script only supports Linux systems.")
            sys.exit(1)
        
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
    
    def check_llama_cpp_binaries(self) -> bool:
        """Check if llama.cpp binaries are available (precompiled or built)"""
        required_binaries = ['llama-cli', 'llama-server']
        
        # Check for binaries in llama.cpp directory (primary location)
        for binary in required_binaries:
            binary_path = Path(self.config['llama_cpp_dir']) / binary
            if binary_path.exists():
                return True
        
        # Check for binaries in llama.cpp/build/bin directory (CUDA build location)
        for binary in required_binaries:
            binary_path = Path(self.config['llama_cpp_dir']) / 'build' / 'bin' / binary
            if binary_path.exists():
                return True
        
        # Check for precompiled binaries in current directory (fallback)
        for binary in required_binaries:
            if Path(binary).exists():
                return True
        
        # Check if binaries are in PATH (fallback)
        for binary in required_binaries:
            if self.check_command_exists(binary):
                return True
        
        return False
    
    def download_precompiled_binaries(self) -> bool:
        """Precompiled binary downloading is disabled - always build from source"""
        self.print_status("Precompiled binary downloading is disabled")
        self.print_status("Always building from source for optimal GPU support")
        return False
    
    def _move_binaries_to_llama_dir(self):
        """Move downloaded binaries to llama.cpp directory and set permissions"""
        # Create llama.cpp directory if it doesn't exist
        os.makedirs(self.config['llama_cpp_dir'], exist_ok=True)
        
        # Look for binaries in current directory and subdirectories
        binaries_to_move = ['llama-cli', 'llama-server', 'llama-quantize', 'llama-gguf-split', 'llama-mtmd-cli']
        
        for binary in binaries_to_move:
            moved = False
            
            # Check if binary exists in current directory
            if os.path.exists(binary):
                src = binary
                dst = os.path.join(self.config['llama_cpp_dir'], binary)
                shutil.move(src, dst)
                self.print_status(f"Moved {binary} to {self.config['llama_cpp_dir']}/")
                moved = True
            
            # Check if binary exists in common subdirectories
            if not moved:
                for subdir in ['bin', 'build/bin', '.']:
                    subdir_path = os.path.join(subdir, binary)
                    if os.path.exists(subdir_path):
                        src = subdir_path
                        dst = os.path.join(self.config['llama_cpp_dir'], binary)
                        shutil.move(src, dst)
                        self.print_status(f"Moved {src} to {self.config['llama_cpp_dir']}/")
                        moved = True
                        break
            
            # Set execute permissions if we moved binary
            if moved:
                dst_path = os.path.join(self.config['llama_cpp_dir'], binary)
                try:
                    os.chmod(dst_path, 0o755)  # rwxr-xr-x
                    self.print_status(f"Set execute permissions for {binary}")
                except OSError as e:
                    self.print_warning(f"Failed to set permissions for {binary}: {e}")
        
        # Also move shared libraries if they exist
        self._move_shared_libraries()
    
    def _fix_binary_permissions(self):
        """Fix permissions for existing binaries in llama.cpp directory"""
        binaries = ['llama-cli', 'llama-server', 'llama-quantize', 'llama-gguf-split', 'llama-mtmd-cli']
        
        for binary in binaries:
            binary_path = os.path.join(self.config['llama_cpp_dir'], binary)
            if os.path.exists(binary_path):
                try:
                    current_mode = os.stat(binary_path).st_mode
                    if not (current_mode & 0o111):  # Check if execute bit is missing
                        os.chmod(binary_path, 0o755)
                        self.print_status(f"Fixed execute permissions for {binary}")
                except OSError as e:
                    self.print_warning(f"Failed to fix permissions for {binary}: {e}")
    
    def _move_shared_libraries(self):
        """Move shared libraries to llama.cpp directory"""
        shared_libs = ['libllama.so', 'libggml.so', 'libggml-base.so', 'libggml-cpu.so', 'libggml-metal.so']
        
        # Look for shared libraries in current directory and subdirectories
        for lib in shared_libs:
            lib_found = False
            
            # Check current directory
            if os.path.exists(lib):
                src = lib
                dst = os.path.join(self.config['llama_cpp_dir'], lib)
                shutil.move(src, dst)
                self.print_status(f"Moved {lib} to {self.config['llama_cpp_dir']}/")
                lib_found = True
            
            # Check common subdirectories (expanded search)
            if not lib_found:
                search_dirs = ['lib', 'build/lib', 'build/src', 'build', 'bin', '.']
                for subdir in search_dirs:
                    lib_path = os.path.join(subdir, lib)
                    if os.path.exists(lib_path):
                        src = lib_path
                        dst = os.path.join(self.config['llama_cpp_dir'], lib)
                        shutil.move(src, dst)
                        self.print_status(f"Moved {src} to {self.config['llama_cpp_dir']}/")
                        lib_found = True
                        break
            
            # If still not found, do a broader search
            if not lib_found:
                for root, dirs, files in os.walk('.'):
                    if lib in files:
                        src = os.path.join(root, lib)
                        dst = os.path.join(self.config['llama_cpp_dir'], lib)
                        shutil.move(src, dst)
                        self.print_status(f"Found and moved {src} to {self.config['llama_cpp_dir']}/")
                        lib_found = True
                        break
    
    def _set_ld_library_path(self):
        """Set LD_LIBRARY_PATH to include llama.cpp directory"""
        llama_cpp_dir = os.path.abspath(self.config['llama_cpp_dir'])
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        
        if llama_cpp_dir not in current_ld_path:
            new_ld_path = f"{llama_cpp_dir}:{current_ld_path}" if current_ld_path else llama_cpp_dir
            os.environ['LD_LIBRARY_PATH'] = new_ld_path
            self.print_status(f"Set LD_LIBRARY_PATH to include {llama_cpp_dir}")
    
    def _check_shared_libraries(self) -> bool:
        """Check if required shared libraries are available"""
        required_libs = ['libllama.so']
        llama_cpp_dir = self.config['llama_cpp_dir']
        
        missing_libs = []
        found_libs = []
        
        for lib in required_libs:
            lib_path = os.path.join(llama_cpp_dir, lib)
            if os.path.exists(lib_path):
                found_libs.append(lib)
            else:
                missing_libs.append(lib)
        
        if missing_libs:
            self.print_status(f"Missing libraries: {', '.join(missing_libs)}")
            self.print_status(f"Found libraries: {', '.join(found_libs)}")
            return False
        else:
            self.print_status(f"All required libraries found: {', '.join(found_libs)}")
            return True
    

    
    def _rebuild_from_source(self):
        """Rebuild from source to get missing shared libraries"""
        # Never remove existing binaries - just rebuild
        self._build_from_source_linux()
    
    def build_llama_cpp(self):
        """Build llama.cpp from source on Linux only"""
        system = self.hardware.system_info['platform']
        
        if system != 'Linux':
            self.print_error("This script only supports Linux systems")
            sys.exit(1)
        
        # Check if binaries already exist before building
        if self.check_llama_cpp_binaries():
            self.print_success("llama.cpp binaries already available")
            # Fix permissions for existing binaries
            self._fix_binary_permissions()
            # Check shared libraries but don't force rebuild unless critical
            if not self._check_shared_libraries():
                self.print_warning("Some shared libraries missing, but binaries may still work")
                # Don't automatically rebuild - let the user decide if needed
            return
        
        # Build from source on Linux
        self._build_from_source_linux()
    
    def _build_from_source_linux(self):
        """Build llama.cpp from source on Linux with CUDA support"""
        # Never remove existing llama.cpp directory - preserve existing builds
        
        # Clone and build from source
        self.print_status("Building llama.cpp from source on Linux...")
        
        try:
            # Clone the repository with shallow clone for speed
            if not Path('llama.cpp').exists():
                self.print_status("Cloning llama.cpp repository (shallow clone for speed)...")
                subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/ggerganov/llama.cpp.git'], check=True)
            else:
                # Update existing repository
                self.print_status("Updating existing llama.cpp repository...")
                os.chdir('llama.cpp')
                subprocess.run(['git', 'fetch', '--depth', '1'], check=True)
                subprocess.run(['git', 'reset', '--hard', 'origin/main'], check=True)
                os.chdir('..')
            
            # Build with appropriate flags for Linux
            os.chdir('llama.cpp')
            
            # Optimized cmake configuration for faster builds
            cmake_args = ['cmake', '-B', 'build', '-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=ON']
            
            # Linux build with appropriate GPU support
            if self.hardware.gpu_info['nvidia_available']:
                self.print_status("Building optimized CUDA version for NVIDIA GPUs...")
                cmake_args.extend(['-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=native'])
                # Additional CUDA optimizations
                cmake_args.extend(['-DGGML_CUDA_F16=ON', '-DGGML_CUDA_DMMV=ON', '-DGGML_CUDA_MMQ=ON'])
            elif self.hardware.gpu_info['amd_available']:
                self.print_status("Building optimized ROCm version for AMD GPUs...")
                cmake_args.extend(['-DGGML_HIPBLAS=ON'])
            else:
                self.print_error("No supported GPU detected. This script requires NVIDIA or AMD GPU support.")
                self.print_status("CPU-only builds are not supported.")
                sys.exit(1)
            
            # Add parallel compilation flags
            cmake_args.extend(['-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG', '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG'])
            
            # Configure build
            subprocess.run(cmake_args, check=True)
            
            # Build with optimizations for speed
            build_cmd = ['cmake', '--build', 'build', '--config', 'Release', '-j', str(self.hardware.cpu_info['logical_cores'])]
            
            # Add optimization flags for faster compilation
            build_cmd.extend(['--', '-l', str(self.hardware.cpu_info['logical_cores'])])
            
            self.print_status(f"Building with {self.hardware.cpu_info['logical_cores']} parallel jobs...")
            subprocess.run(build_cmd, check=True)
            
            # Copy binaries to expected location
            os.chdir('..')
            os.makedirs(self.config['llama_cpp_dir'], exist_ok=True)
            
            # Copy built binaries
            binaries_to_copy = ['llama-cli', 'llama-server', 'llama-quantize', 'llama-gguf-split']
            build_dir = Path('llama.cpp/build/bin')
            
            for binary in binaries_to_copy:
                src_path = build_dir / binary
                dst_path = Path(self.config['llama_cpp_dir']) / binary
                
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    os.chmod(dst_path, 0o755)
                    self.print_status(f"Copied {binary} to {self.config['llama_cpp_dir']}/")
                else:
                    self.print_warning(f"Binary {binary} not found in build output")
            
            # Copy shared libraries if they exist
            lib_build_dir = Path('llama.cpp/build/lib')
            if lib_build_dir.exists():
                for lib_file in lib_build_dir.glob('*.so*'):
                    dst_path = Path(self.config['llama_cpp_dir']) / lib_file.name
                    shutil.copy2(lib_file, dst_path)
                    self.print_status(f"Copied library {lib_file.name}")
            
            # Copy CUDA libraries if they exist
            cuda_lib_build_dir = Path('llama.cpp/build/bin')
            if cuda_lib_build_dir.exists():
                for lib_file in cuda_lib_build_dir.glob('libggml-cuda.so*'):
                    dst_path = Path(self.config['llama_cpp_dir']) / lib_file.name
                    shutil.copy2(lib_file, dst_path)
                    self.print_status(f"Copied CUDA library {lib_file.name}")
                for lib_file in cuda_lib_build_dir.glob('libmtmd.so*'):
                    dst_path = Path(self.config['llama_cpp_dir']) / lib_file.name
                    shutil.copy2(lib_file, dst_path)
                    self.print_status(f"Copied MTMD library {lib_file.name}")
            
            self.print_success("llama.cpp built successfully from source on Linux")
            
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to build llama.cpp on Linux: {e}")
            self.print_error("Build failed - cannot proceed without compiled binaries")
            sys.exit(1)
        
        finally:
            # Return to original directory
            os.chdir('..') if os.getcwd().endswith('llama.cpp') else None
    
    def _download_precompiled_binaries_fallback(self):
        """Fallback method - always build from source"""
        if self.check_llama_cpp_binaries():
            self.print_success("llama.cpp binaries already available")
            # Fix permissions for existing binaries
            self._fix_binary_permissions()
            # Check shared libraries but don't force rebuild
            if not self._check_shared_libraries():
                self.print_warning("Some shared libraries missing, but binaries may still work")
            return
        
        # Always build from source for optimal GPU support
        self.print_status("Building llama.cpp from source for optimal GPU support")
        self._build_from_source_linux()
        
        # For NVIDIA GPUs, always build from source for CUDA support
        if self.hardware.gpu_info['nvidia_available']:
            self.print_error("NVIDIA GPU detected but no CUDA binaries found - building from source")
            self._build_from_source_linux()
            return
        
        # For AMD GPUs, build from source for ROCm support
        if self.hardware.gpu_info['amd_available']:
            self.print_error("AMD GPU detected but no ROCm binaries found - building from source")
            self._build_from_source_linux()
            return
        
        # Verify shared libraries are available
        if not self._check_shared_libraries():
            self.print_error("Shared libraries missing after download")
            sys.exit(1)
    
    def get_available_quantizations(self) -> List[str]:
        """Get list of available quantizations in the repository"""
        try:
            # First try to check if repository is accessible
            from huggingface_hub import model_info
            try:
                info = model_info(self.config['model_repo'])
                self.print_status(f"Repository accessible: {self.config['model_repo']}")
                self.print_status(f"Model files: {len(info.siblings) if info.siblings else 0}")
            except Exception as repo_error:
                self.print_error(f"Repository not accessible: {repo_error}")
                return self._get_fallback_quantizations()
            
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem()
            all_files = fs.glob(f"{self.config['model_repo']}/*")
            gguf_files = [f for f in all_files if f.endswith('.gguf')]
            
            self.print_status(f"Found {len(gguf_files)} GGUF files in repository")
            
            # Based on Unsloth documentation, available quantizations are:
            available_quants = set()
            target_quants = ["UD-TQ1_0", "UD-IQ1_S", "UD-IQ1_M", "UD-IQ2_XXS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-Q4_K_XL", "UD-Q5_K_XL"]
            
            # Check for directories containing quantization files
            for f in all_files:
                if f.endswith('/'):
                    # It's a directory, check if it matches quantization pattern
                    dir_name = f.split('/')[-2] if f.endswith('/') else f.split('/')[-1]
                    for quant in target_quants:
                        if quant in dir_name:
                            available_quants.add(quant)
            
            # Also check individual GGUF files
            for f in gguf_files:
                for quant in target_quants:
                    if quant in f:
                        available_quants.add(quant)
            
            # Additional check: look for all files and directories more thoroughly
            self.print_status("Debug: Scanning all repository contents...")
            for f in all_files:
                # Check if it's a directory path
                if '/' in f and not f.endswith('.gguf'):
                    # Extract directory name
                    path_parts = f.split('/')
                    if len(path_parts) >= 2:
                        dir_name = path_parts[-2]  # Second to last part is directory name
                        for quant in target_quants:
                            if quant in dir_name:
                                available_quants.add(quant)
                                self.print_status(f"Found quantization directory: {dir_name}")
                
                # Check if it's a GGUF file
                if f.endswith('.gguf'):
                    for quant in target_quants:
                        if quant in f:
                            available_quants.add(quant)
                            self.print_status(f"Found quantization file: {f.split('/')[-1]}")
            
            if not available_quants:
                # Debug: show some file names and directories
                self.print_status("Debug: First few GGUF files found:")
                for i, f in enumerate(gguf_files[:5]):
                    self.print_status(f"  {f}")
                
                self.print_status("Debug: Directories found:")
                dirs = [f for f in all_files if f.endswith('/')]
                for i, d in enumerate(dirs[:10]):
                    dir_name = d.split('/')[-2] if d.endswith('/') else d.split('/')[-1]
                    self.print_status(f"  {dir_name}")
                
                # Fallback to known available quantizations from documentation
                self.print_status("Using fallback quantization list from documentation")
                return self._get_fallback_quantizations()
            
            return sorted(list(available_quants))
            
        except Exception as e:
            self.print_error(f"Failed to get quantizations: {e}")
            return self._get_fallback_quantizations()
    
    def _get_fallback_quantizations(self) -> List[str]:
        """Get fallback quantization list from documentation"""
        return ["UD-TQ1_0", "UD-IQ1_S", "UD-IQ1_M", "UD-IQ2_XXS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-Q4_K_XL", "UD-Q5_K_XL"]
    
    def get_best_available_quantization(self) -> str:
        """Get the best available quantization based on hardware and availability"""
        available_quants = self.get_available_quantizations()
        
        if not available_quants:
            self.print_error("No quantizations available in repository")
            sys.exit(1)
        
        # Priority order based on quality
        priority_order = [
            "UD-Q5_K_XL", "UD-Q4_K_XL", "UD-Q3_K_XL", "UD-IQ3_XXS",
            "UD-Q2_K_XL", "UD-IQ2_XXS", "UD-IQ1_M", "UD-IQ1_S", "UD-TQ1_0"
        ]
        
        # Find the best available quantization
        for quant in priority_order:
            if quant in available_quants:
                return quant
        
        # Fallback to first available
        return available_quants[0]
    
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
        
        if parallel:
            self.print_status(f"[PARALLEL] Checking for existing model files with patterns: {model_patterns}")
        
        # Check if model files already exist (recursive search)
        existing_files = []
        for pattern in model_patterns:
            files = list(model_dir.rglob(pattern))  # Recursive search
            existing_files.extend(files)
            if parallel and files:
                self.print_status(f"[PARALLEL] Found {len(files)} files matching {pattern}")
        
        if existing_files:
            total_size = sum(f.stat().st_size for f in existing_files)
            size_gb = total_size / (1024**3)
            if parallel:
                self.print_status(f"[PARALLEL] Model files already exist ({size_gb:.1f}GB total) - skipping download")
            else:
                self.print_success(f"Optimized model files already exist ({size_gb:.1f}GB total)")
                # Show found files
                for f in existing_files:
                    size_gb = f.stat().st_size / (1024**3)
                    self.print_status(f"  Found: {f.name} ({size_gb:.1f}GB)")
            return True
        
        if not parallel:
            self.print_status(f"Downloading optimized GLM-4.6 model ({self.config['quant_type']} quantization)...")
        
        # Create simplified download script with better debugging
        download_script = f'''
import os
import sys
import time
import threading
from huggingface_hub import snapshot_download, hf_hub_download
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

repo_id = "{self.config['model_repo']}"
local_dir = "{self.config['model_dir']}"
quant_type = "{self.config['quant_type']}"

print(f"DEBUG: Using quantization: {{quant_type}}")

print(f"Downloading optimized {{quant_type}} model files...")

try:
    # Ensure local directory exists
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    print(f"Download directory: {{Path(local_dir).absolute()}}")
    
    # First, list available files in the repo
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    all_files = fs.glob(f"{{repo_id}}/*")
    
    # Look for both GGUF files and directories
    gguf_files = [f for f in all_files if f.endswith('.gguf')]
    directories = [f for f in all_files if f.endswith('/')]
    
    print(f"Found {{len(gguf_files)}} GGUF files and {{len(directories)}} directories in repository:")
    
    # Show GGUF files
    for f in gguf_files:
        if quant_type in f:
            print(f"  - GGUF: {{f.split('/')[-1]}}")
    
    # Show directories that might contain quantization files
    for d in directories:
        dir_name = d.split('/')[-2] if d.endswith('/') else d.split('/')[-1]
        if quant_type in dir_name:
            print(f"  - Directory: {{dir_name}}")
    
    # Filter files for our quantization (both direct GGUF files and directories)
    target_files = []
    is_direct_file = False
    
    # First check for direct GGUF file with exact quantization match
    for f in gguf_files:
        filename = f.split('/')[-1]
        # Look for files that start with "GLM-4.6-" and contain the quantization
        if filename.startswith("GLM-4.6-") and quant_type in filename:
            target_files.append(f)  # Store the full path from fs.glob()
            is_direct_file = True
            print(f"Found direct file: {{filename}}")
    
    # If no direct file found, look in directories
    if not is_direct_file:
        for d in directories:
            dir_name = d.split('/')[-2] if d.endswith('/') else d.split('/')[-1]
            if quant_type in dir_name:
                # Add all GGUF files in this directory
                dir_files = fs.glob(f"{{repo_id}}/{{dir_name}}/*.gguf")
                target_files.extend(dir_files)
                print(f"Found directory with {{len(dir_files)}} files: {{dir_name}}")
    
    if not target_files:
        print(f"No files found matching quantization: {{quant_type}}")
        print("Available quantizations:")
        quants = set()
        for f in gguf_files:
            for q in ["UD-TQ1_0", "UD-IQ1_S", "UD-IQ1_M", "UD-IQ2_XXS", "UD-Q2_K_XL", "UD-IQ3_XXS", "UD-Q3_K_XL", "UD-Q4_K_XL", "UD-Q5_K_XL"]:
                if q in f:
                    quants.add(q)
        for q in sorted(quants):
            print(f"  - {{q}}")
        sys.exit(1)
    
    print(f"Downloading {{len(target_files)}} files...")
    
    # Download each file individually for better control
    downloaded_files = []
    for file_path in target_files:
        # Extract just the filename from the full path
        filename = file_path.split('/')[-1]
        
        # Debug: show what we're working with
        print(f"DEBUG: file_path = '{{file_path}}', filename = '{{filename}}'")
        print(f"DEBUG: target_files contains: {{target_files}}")
        
        # Check if this is a direct file (no subdirectory)
        # Direct files will have path like "unsloth/GLM-4.6-GGUF/GLM-4.6-UD-TQ1_0.gguf" 
        # Files in subdirectories will have path like "unsloth/GLM-4.6-GGUF/UD-Q4_K_XL/somefile.gguf"
        path_parts = file_path.split('/')
        
        # If we have exactly 3 parts and the middle part is the repo name, it's a direct file
        # If we have 4+ parts, it's in a subdirectory
        if len(path_parts) == 3 and path_parts[1] == repo_id.split('/')[-1]:
            # Direct file at repository root
            download_filename = filename
            print(f"Downloading direct file: {{filename}}")
        else:
            # File in subdirectory
            # Extract subdirectory path (everything after repo name, before filename)
            if len(path_parts) >= 4:
                subdirectory = '/'.join(path_parts[2:-1])  # Skip repo name, exclude filename
                download_filename = f"{{subdirectory}}/{{filename}}"
                print(f"Downloading from subdirectory: {{subdirectory}}/{{filename}}")
            else:
                download_filename = filename
                print(f"Downloading file: {{filename}}")
        
        print(f"DEBUG: download_filename = '{{download_filename}}'")
        
        local_file_path = Path(local_dir) / filename
        
        hf_hub_download(
            repo_id=repo_id,
            filename=download_filename,
            local_dir=local_dir
        )
        
        downloaded_files.append(local_file_path)
        
        # Show file size
        if local_file_path.exists():
            size_gb = local_file_path.stat().st_size / (1024**3)
            print(f"  ✓ {{filename}} ({{size_gb:.1f}}GB)")
        else:
            print(f"  ✗ {{filename}} (FAILED)")
    
    # Verify all files were downloaded
    total_size_gb = sum(f.stat().st_size for f in downloaded_files if f.exists()) / (1024**3)
    successful_files = [f for f in downloaded_files if f.exists()]
    
    print(f"\\nDownload completed! {{len(successful_files)}}/{{len(target_files)}} files ({{total_size_gb:.1f}}GB total):")
    for f in successful_files:
        size_gb = f.stat().st_size / (1024**3)
        print(f"  - {{f.name}} ({{size_gb:.1f}}GB)")
    
    if len(successful_files) < len(target_files):
        print(f"\\nWARNING: {{len(target_files) - len(successful_files)}} files failed to download")
        sys.exit(1)
    
    print("Optimized model download completed successfully!")
        
except Exception as e:
    print(f"\\nDownload failed: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
        
        with open('download_model.py', 'w') as f:
            f.write(download_script)
        
        # Set up environment for subprocess
        env = os.environ.copy()
        
        # Use the same Python executable that's running this script
        python_exe = sys.executable
        
        # Find the correct site-packages directory
        import site
        user_site = site.getusersitepackages()
        
        # Add current directory and user site-packages to PYTHONPATH
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env['PYTHONPATH'] = f"{current_dir}:{user_site}:{env.get('PYTHONPATH', '')}"
        
        # Debug info
        self.print_status(f"Using Python: {python_exe}")
        self.print_status(f"User site-packages: {user_site}")
        self.print_status(f"PYTHONPATH: {env['PYTHONPATH']}")
        
        if parallel:
            # Run in background thread
            def download_worker():
                try:
                    # Add debug output
                    print(f"[PARALLEL DEBUG] Starting download script: {sys.executable} download_model.py")
                    
                    result = subprocess.run([sys.executable, 'download_model.py'], 
                                         check=True, capture_output=True, text=True, 
                                         cwd=os.getcwd(), env=env)
                    
                    print(f"[PARALLEL DEBUG] Script completed with return code: {result.returncode}")
                    
                    if result.stdout:
                        print(f"[PARALLEL STDOUT]:")
                        for line in result.stdout.strip().split('\n'):
                            if line.strip():
                                print(f"  {line}")
                    
                    if result.stderr:
                        print(f"[PARALLEL STDERR]:")
                        for line in result.stderr.strip().split('\n'):
                            if line.strip():
                                print(f"  {line}")
                                
                except subprocess.CalledProcessError as e:
                    print(f"[PARALLEL ERROR] Download failed with exit code {e.returncode}")
                    if e.stdout:
                        print(f"[PARALLEL STDOUT]: {e.stdout}")
                    if e.stderr:
                        print(f"[PARALLEL STDERR]: {e.stderr}")
                except Exception as e:
                    print(f"[PARALLEL ERROR] Unexpected error: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    if os.path.exists('download_model.py'):
                        os.remove('download_model.py')
                        print("[PARALLEL DEBUG] Cleaned up download_model.py")
            
            thread = threading.Thread(target=download_worker, daemon=True)
            thread.start()
            print(f"[PARALLEL DEBUG] Thread started: {thread.name}")
            return thread
        else:
            # Use the same Python executable and import modules directly
            python_exe = sys.executable
            
            # Debug info
            self.print_status(f"Using Python: {python_exe}")
            
            # Try to import huggingface_hub in current process first
            try:
                import huggingface_hub
                self.print_status("huggingface_hub is available in main process")
            except ImportError as e:
                self.print_error(f"huggingface_hub not available: {e}")
                self.print_status("Installing huggingface_hub...")
                subprocess.run([python_exe, '-m', 'pip', 'install', '--user', 'huggingface_hub'], check=True)
            
            subprocess.run([python_exe, 'download_model.py'], check=True, env=env)
            os.remove('download_model.py')
            self.print_success("Optimized model download completed")
            return True
    
    def merge_model_files(self) -> str:
        """Merge split GGUF files if necessary"""
        self.print_status("Checking if optimized model files need merging...")
        
        model_dir = Path(self.config['model_dir'])
        model_patterns = self.get_optimized_model_files()
        
        # Check for split files across all patterns (recursive search)
        split_files = []
        for pattern in model_patterns:
            split_files.extend(model_dir.rglob(f"{pattern.replace('*', '')}-00001-of-*.gguf"))
        
        if split_files:
            self.print_status("Merging split GGUF files...")
            first_file = split_files[0]
            merged_file = model_dir / f"GLM-4.6-{self.config['quant_type']}-merged.gguf"
            
            gguf_split_path = self.get_binary_path('llama-gguf-split')
            subprocess.run([
                gguf_split_path,
                '--merge', str(first_file), str(merged_file)
            ], check=True)
            
            model_path = str(merged_file)
            self.print_success("Optimized model files merged successfully")
        else:
            # Find the main optimized model file (recursive search)
            model_files = []
            for pattern in model_patterns:
                model_files.extend(model_dir.rglob(pattern))
            
            # First, look for split files and use the first one
            split_files = [f for f in model_files if '00001-of-' in f.name]
            if split_files:
                model_path = str(split_files[0])
                self.print_status(f"Using split model file: {Path(model_path).name}")
            else:
                # Filter out split files to find single files
                single_files = [f for f in model_files if '-0000' not in f.name]
                
                if single_files:
                    # Choose the largest file (usually the main model)
                    single_files.sort(key=lambda x: x.stat().st_size, reverse=True)
                    model_path = str(single_files[0])
                    self.print_status(f"Using largest model file: {Path(model_path).name}")
                else:
                    # Fallback to any matching file
                    for pattern in model_patterns:
                        fallback_files = list(model_dir.rglob(pattern))
                        if fallback_files:
                            model_path = str(fallback_files[0])
                            break
                    else:
                        model_path = ""
            
            if model_path:
                self.print_success(f"Using optimized model file: {Path(model_path).name}")
            else:
                self.print_error("No model files found!")
                self.print_status("Debugging - checking directory structure:")
                if model_dir.exists():
                    for item in model_dir.rglob("*"):
                        if item.is_file():
                            size_gb = item.stat().st_size / (1024**3)
                            self.print_status(f"  Found file: {item.relative_to(model_dir)} ({size_gb:.1f}GB)")
                sys.exit(1)
        
        return model_path
    
    def get_binary_path(self, binary_name: str) -> str:
        """Get the correct path for a llama.cpp binary"""
        # Check llama.cpp directory first
        llama_cpp_path = Path(self.config['llama_cpp_dir']) / binary_name
        if llama_cpp_path.exists():
            return str(llama_cpp_path)
        
        # Check llama.cpp/build/bin directory (CUDA build location)
        build_path = Path(self.config['llama_cpp_dir']) / 'build' / 'bin' / binary_name
        if build_path.exists():
            return str(build_path)
        
        # Check current directory
        current_path = Path(binary_name)
        if current_path.exists():
            return binary_name
        
        # Check PATH
        if self.check_command_exists(binary_name):
            return binary_name
        
        # Default to llama.cpp directory path
        return f'./{self.config["llama_cpp_dir"]}/{binary_name}'
    
    def check_gpu_support(self) -> bool:
        """Check if the llama.cpp binary supports GPU"""
        try:
            cli_path = self.get_binary_path('llama-cli')
            result = subprocess.run([cli_path, '--help'], capture_output=True, text=True, timeout=10)
            help_text = result.stdout + result.stderr
            return '--n-gpu-layers' in help_text
        except:
            return False
    
    def get_available_buffer_types(self) -> list:
        """Get available buffer types from the binary"""
        try:
            cli_path = self.get_binary_path('llama-cli')
            result = subprocess.run([cli_path, '--help'], capture_output=True, text=True, timeout=10)
            help_text = result.stdout + result.stderr
            
            # Look for buffer types in help output
            if 'Available buffer types:' in help_text:
                buffer_section = help_text.split('Available buffer types:')[1].split('\n')[0]
                return [bt.strip() for bt in buffer_section.split() if bt.strip()]
            return []
        except:
            return []
    
    def build_command(self, model_path: str, server_mode: bool = False) -> List[str]:
        """Build optimized command based on hardware"""
        if server_mode:
            binary_path = self.get_binary_path('llama-server')
            cmd = [
                binary_path,
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
            binary_path = self.get_binary_path('llama-cli')
            cmd = [
                binary_path,
                '--model', model_path,
                '--jinja',
                '--ctx-size', str(self.config['context_size']),
                '--temp', str(self.config['temperature']),
                '--top-p', str(self.config['top_p']),
                '--top-k', str(self.config['top_k']),
                '--seed', str(self.config['seed']),
                '--color', '-i'
            ]
        
        # Check if binary supports GPU and available buffer types
        gpu_support = self.check_gpu_support()
        buffer_types = self.get_available_buffer_types()
        
        # Add hardware-specific optimizations
        if self.hardware.gpu_info['nvidia_available'] and gpu_support:
            cmd.extend(['--n-gpu-layers', str(self.optimal_settings['gpu_layers'])])
            
            # Always use CUDA for NVIDIA GPUs on Linux
            if self.hardware.gpu_info['nvidia_available']:
                self.print_status("Using CUDA GPU backend for NVIDIA GPUs")
                is_cuda = True
            else:
                self.print_status("Using CPU backend (no NVIDIA GPU detected)")
                is_cuda = False
            
            if is_cuda:
                # CUDA binary - can use advanced features
                # Only add tensor override if we have valid buffer types
                if 'f32' in buffer_types:
                    cmd.extend(['-ot', '.ffn_.*_exps.=f32'])
                
                if self.optimal_settings['cache_quantization'] and 'q4_1' in buffer_types:
                    cmd.extend(['--cache-type-k', 'q4_1', '--cache-type-v', 'q4_1'])
                
                if self.optimal_settings['use_flash_attention']:
                    cmd.extend(['--flash-attn', 'on'])
                
                # High-end GPU optimizations for H200/H100
                if self.hardware.gpu_info['gpu_memory'] >= 80000:
                    if 'f16' in buffer_types:
                        cmd.extend(['-ot', '.*=f16'])
                    # Add parallel processing optimizations
                    cmd.extend(['--parallel', str(self.hardware.cpu_info['logical_cores'])])
                    # Optimize batch size for H200 performance
                    cmd.extend(['--batch-size', '1024'])  # Further reduced for better latency
                    # GPU layers already set in base command
                    # Enable tensor parallelism if available
                    if self.hardware.gpu_info['gpu_count'] > 1:
                        cmd.extend(['--tensor-split', f"{self.hardware.gpu_info['gpu_memory']//2},{self.hardware.gpu_info['gpu_memory']//2}"])
                    # Additional H200 optimizations
                    cmd.extend(['--main-gpu', '0'])  # Use primary GPU for main operations
                    if 'f32' in buffer_types:
                        cmd.extend(['-ot', 'attn_output=f32'])  # Keep attention output in f32 for accuracy
        
        elif self.hardware.gpu_info['apple_silicon'] and gpu_support:
            cmd.extend(['--n-gpu-layers', str(self.optimal_settings['gpu_layers'])])
            if self.optimal_settings['use_flash_attention']:
                cmd.extend(['--flash-attn', 'on'])
        
        else:
            # CPU-only optimization - no GPU-specific arguments
            cmd.extend(['--threads', str(self.optimal_settings['threads'])])
            if not gpu_support and (self.hardware.gpu_info['nvidia_available'] or self.hardware.gpu_info['apple_silicon']):
                self.print_warning("GPU detected but binary doesn't support GPU - running in CPU mode")
        
        # Add universal performance optimizations
        if not server_mode:
            # Interactive mode optimizations
            if not (self.hardware.gpu_info['nvidia_available'] and self.hardware.gpu_info['gpu_memory'] >= 80000):
                cmd.extend(['--batch-size', str(self.optimal_settings['batch_size'])])
            cmd.extend(['--keep', '0'])  # Don't keep prompt in context
            # Context size already set in base command
            # Enable low-memory mode for better performance
            if self.hardware.gpu_info['gpu_memory'] >= 80000:
                cmd.extend(['--grp-attn-n', '1'])  # Grouped attention optimization
                cmd.extend(['--p-conv', '1'])  # Enable position convolution for better performance
                cmd.extend(['--rope-scaling', 'none'])  # Disable rope scaling for speed
        
        # Memory optimizations for large context
        if self.config['context_size'] >= 100000:
            # Note: --memory-f16 flag may not be available in all builds
            # Only add if binary supports it
            try:
                result = subprocess.run([cmd[0], '--help'], capture_output=True, text=True, timeout=5)
                if '--memory-f16' in result.stdout + result.stderr:
                    cmd.extend(['--memory-f16'])  # Use half precision for KV cache if available
            except:
                pass
        
        # Performance optimization info
        if buffer_types:
            self.print_status(f"Available buffer types: {', '.join(buffer_types)}")
        else:
            self.print_status("Buffer type information not available")
        
        # Show performance optimizations being applied
        if self.hardware.gpu_info['nvidia_available'] and gpu_support:
            self.print_status(f"GPU optimizations: {self.optimal_settings['gpu_layers']} layers, batch_size={self.optimal_settings['batch_size']}")
            if self.hardware.gpu_info['gpu_memory'] >= 80000:
                self.print_status("High-end GPU optimizations enabled (H200/H100 class)")
            if self.optimal_settings.get('use_tensor_cores'):
                self.print_status("Tensor cores optimization enabled")
        else:
            self.print_status(f"CPU optimizations: {self.optimal_settings['threads']} threads, batch_size={self.optimal_settings['batch_size']}")
        
        # Filter out potentially problematic arguments for CPU-only builds
        if not gpu_support:
            filtered_cmd = []
            i = 0
            while i < len(cmd):
                arg = cmd[i]
                # Skip GPU-specific arguments
                if arg == '--n-gpu-layers' or arg == '--flash-attn':
                    # Skip this argument and its value if it has one
                    if i + 1 < len(cmd) and not cmd[i + 1].startswith('-'):
                        i += 1
                elif arg == '-ot' or arg.startswith('--cache-type-'):
                    # Skip this argument and its value if it has one
                    if i + 1 < len(cmd) and not cmd[i + 1].startswith('-'):
                        i += 1
                elif arg.startswith('-ot='):
                    # Skip combined form
                    pass
                else:
                    filtered_cmd.append(arg)
                i += 1
            cmd = filtered_cmd
        
        return cmd
    
    def run_model(self, model_path: str):
        """Run GLM-4.6 in interactive mode"""
        self.print_status("Starting GLM-4.6 with 200K context...")
        
        os.environ['LLAMA_CACHE'] = self.config['model_dir']
        
        # Set library path for shared libraries
        self._set_ld_library_path()
        
        cmd = self.build_command(model_path, server_mode=False)
        
        self.print_status("Running with command:")
        print(f"{Colors.YELLOW}{' '.join(cmd)}{Colors.NC}")
        self.print_status("Note: First load may take several minutes for 200K context initialization...")
        
        # Check if we're in an interactive terminal
        if not sys.stdin.isatty():
            self.print_warning("Not in interactive terminal. Starting server mode instead...")
            self.run_server(model_path)
            return
        
        # Run interactive mode
        try:
            self.print_status("GLM-4.6 is ready! Starting interactive mode...")
            self.print_status("Press Ctrl+C to exit")
            print("")
            
            # Run with proper input handling
            process = subprocess.Popen(cmd, text=True)
            process.wait()
            
        except KeyboardInterrupt:
            self.print_status("\nExiting GLM-4.6...")
        except Exception as e:
            self.print_error(f"Error running model: {e}")
    
    def run_server(self, model_path: str):
        """Run GLM-4.6 in server mode"""
        self.print_status("Starting GLM-4.6 server with 200K context...")
        
        os.environ['LLAMA_CACHE'] = self.config['model_dir']
        
        # Set library path for shared libraries
        self._set_ld_library_path()
        
        cmd = self.build_command(model_path, server_mode=True)
        
        self.print_status("Starting server with command:")
        print(f"{Colors.YELLOW}{' '.join(cmd)}{Colors.NC}")
        
        subprocess.run(cmd)
    
    def show_usage(self):
        """Show usage information"""
        print("Usage: python run_glm46_200k.py [OPTIONS] (Linux Only)")
        print("")
        print("Options:")
        print("  -h, --help          Show this help message")
        print("  -s, --server        Run in server mode (default: interactive mode)")
        print("  -q, --quant TYPE    Quantization type (default: UD-Q2_K_XL)")
        print("  --skip-deps         Skip dependency installation")
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
        print("")
        print("Note: This script only supports Linux systems with NVIDIA/AMD GPU support.")
    
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
        
        # Monitor download progress by checking file sizes
        model_dir = Path(self.config['model_dir'])
        model_patterns = self.get_optimized_model_files()
        last_size = 0
        
        while download_thread.is_alive():
            # Check current download size
            current_size = 0
            if model_dir.exists():
                for pattern in model_patterns:
                    for f in model_dir.glob(pattern):
                        current_size += f.stat().st_size
            
            # Calculate progress if we have files
            if current_size > 0:
                size_gb = current_size / (1024**3)
                print(f"\r{Colors.YELLOW}[DOWNLOAD]{Colors.NC} {spinner[idx % len(spinner)]} Downloading model... {size_gb:.2f}GB", end='', flush=True)
            else:
                print(f"\r{Colors.YELLOW}[DOWNLOAD]{Colors.NC} {spinner[idx % len(spinner)]} Downloading model... ", end='', flush=True)
            
            idx += 1
            time.sleep(0.5)
            download_thread.join(timeout=0.5)
        
        # Check if thread completed successfully
        if download_thread.is_alive():
            self.print_error("Download thread is still running")
            return False
        
        print(f"\r{Colors.GREEN}[SUCCESS]{Colors.NC} Model download completed!{' ' * 20}")
        
        # Verify files were actually downloaded
        existing_files = []
        for pattern in model_patterns:
            existing_files.extend(model_dir.glob(pattern))
        
        if not existing_files:
            self.print_error("Download completed but no model files found!")
            self.print_status("Checking download directory contents...")
            if model_dir.exists():
                for item in model_dir.iterdir():
                    self.print_status(f"  Found: {item.name}")
            else:
                self.print_status("Model directory does not exist!")
            return False
        
        return True
    
    def main(self):
        """Main execution function"""
        parser = argparse.ArgumentParser(description='GLM-4.6 Runner with 200K Context')
        parser.add_argument('-s', '--server', action='store_true', help='Run in server mode')
        parser.add_argument('-q', '--quant', type=str, default=self.optimal_settings['recommended_quant'],
                          help='Quantization type')
        parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
        parser.add_argument('--skip-download', action='store_true', help='Skip model download')
        parser.add_argument('--hardware-info', action='store_true', help='Show detailed hardware information')
        parser.add_argument('--no-parallel', action='store_true', help='Disable parallel model download')
        parser.add_argument('--debug', action='store_true', help='Enable debug mode to see what\'s being checked')
        
        args = parser.parse_args()
        
        if args.hardware_info:
            self.show_hardware_info()
            return
        
        # Check if requested quantization is available, fallback to best available
        available_quants = self.get_available_quantizations()
        if not available_quants:
            self.print_error("No quantizations available in repository")
            self.print_status("This could be due to:")
            self.print_status("  1. Network connectivity issues")
            self.print_status("  2. Repository not accessible")
            self.print_status("  3. HuggingFace Hub API issues")
            self.print_status("Trying fallback quantization...")
            self.config['quant_type'] = 'UD-Q2_K_XL'  # Fallback to common quantization
        elif args.quant not in available_quants:
            self.print_warning(f"Requested quantization {args.quant} not available")
            self.print_status(f"Available quantizations: {', '.join(available_quants)}")
            self.config['quant_type'] = self.get_best_available_quantization()
            self.print_status(f"Using best available quantization: {self.config['quant_type']}")
        else:
            self.config['quant_type'] = args.quant
        
        self.print_status("GLM-4.6 Runner with 200K Context")
        self.print_status(f"Quantization: {self.config['quant_type']}")
        self.print_status(f"Context Size: {self.config['context_size']}")
        self.print_status(f"Server Mode: {args.server}")
        print("")
        
        # Execute steps
        if not self.check_requirements():
            sys.exit(1)
        
        # Start download (using regular sequential download for better progress display)
        if not args.skip_download:
            self.download_model(parallel=False)
        else:
            self.print_status("Download skipped by user request")
        
        # Install dependencies while model downloads
        if not args.skip_deps:
            self.install_dependencies()
        
        # Download llama.cpp binaries
        self.build_llama_cpp()
        
        # Download is already handled above
        
        model_path = self.merge_model_files()
        
        # Ensure binary permissions are correct before running
        self._fix_binary_permissions()
        
        if args.server:
            self.run_server(model_path)
        else:
            self.run_model(model_path)

if __name__ == '__main__':
    runner = GLMRunner()
    runner.main()