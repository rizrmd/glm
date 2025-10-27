# GLM-4.6 Runner with 200K Context

A Python script to run GLM-4.6 locally with automatic hardware detection and optimization for 200K context support.

## Quick Start

Clone and run immediately:

```bash
git clone https://github.com/rizrmd/glm.git
cd glm
python run_glm46_200k.py
```

The script will automatically:
- Detect your hardware (CPU, GPU, memory, storage)
- Install all required dependencies
- Optimize settings for your system
- Download and configure GLM-4.6 model
- Start the model with optimal parameters

## Features

- 🚀 **Automatic Hardware Detection**: Analyzes CPU cores, GPU type (NVIDIA/AMD/Apple Silicon), memory, and storage
- ⚡ **Dynamic Optimization**: Configures threads, GPU layers, and quantization based on available resources
- 🖥️ **Cross-Platform Support**: Works on Linux and macOS
- 🎯 **Multiple Modes**: Interactive mode and server mode
- 📦 **Dependency Management**: Auto-installs system and Python dependencies
- 🧠 **Full 200K Context**: Optimized for large context windows
- 🔄 **Smart Quantization**: Automatically selects optimal quantization based on hardware

## System Requirements

### Minimum Requirements
- **Disk Space**: 150GB+ (for recommended 2.71bit quantization)
- **RAM**: 32GB+ (recommended for 200K context)
- **CPU**: Multi-core processor
- **Python**: 3.7+

### Recommended Requirements
- **GPU**: NVIDIA GPU with 24GB+ VRAM or Apple Silicon
- **RAM**: 64GB+ 
- **Disk Space**: 200GB+ SSD (for faster loading)

## Usage

```bash
# Show hardware information
python run_glm46_200k.py --hardware-info

# Run interactive mode (default)
python run_glm46_200k.py

# Run server mode
python run_glm46_200k.py --server

# Use specific quantization
python run_glm46_200k.py -q UD-Q4_K_XL

# Skip certain steps
python run_glm46_200k.py --skip-deps --skip-build --skip-download
```

The server will start on `http://localhost:8001` and you can use the Python client:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Interactive mode
python3 glm46_client.py --interactive

# Single query
python3 glm46_client.py --query "What is the meaning of life?"

# Batch processing
python3 glm46_client.py --batch prompts.txt --output results.json
```

## Quantization Options

| Type | Bits | Disk Size | Description |
|------|------|-----------|-------------|
| UD-TQ1_0 | 1.66bit | ~84GB | Smallest size, fastest loading |
| UD-IQ1_S | 1.78bit | ~96GB | Good balance of size and quality |
| UD-IQ1_M | 1.93bit | ~107GB | Better quality than IQ1_S |
| UD-IQ2_XXS | 2.42bit | ~115GB | Good for resource-constrained systems |
| **UD-Q2_K_XL** | **2.71bit** | **~135GB** | **Recommended balance** |
| UD-IQ3_XXS | 3.12bit | ~145GB | Better quality |
| UD-Q3_K_XL | 3.5bit | ~158GB | Good quality/size ratio |
| UD-Q4_K_XL | 4.5bit | ~204GB | High quality |
| UD-Q5_K_XL | 5.5bit | ~252GB | Best quality |

### Using Different Quantizations

```bash
# Use 4.5bit quantization (higher quality, more disk space)
./run_glm46_200k.sh --quant UD-Q4_K_XL

# Use 1.66bit quantization (smallest size)
./run_glm46_200k.sh --quant UD-TQ1_0
```

## Advanced Usage

### Command Line Options

```bash
./run_glm46_200k.sh [OPTIONS]

Options:
  -h, --help          Show help message
  -s, --server        Run in server mode
  -q, --quant TYPE    Quantization type (default: UD-Q2_K_XL)
  --skip-deps         Skip dependency installation
  --skip-build        Skip llama.cpp build
  --skip-download     Skip model download
```

### Memory Optimization for 200K Context

The script automatically applies several optimizations for 200K context:

1. **KV Cache Quantization**: Uses q4_1 quantization for both K and V caches
2. **MoE Layer Offloading**: Offloads MoE expert layers to CPU to save VRAM
3. **Flash Attention**: Enabled for faster processing (when GPU available)

### Manual llama.cpp Execution

If you want to run llama.cpp manually with custom parameters:

```bash
export LLAMA_CACHE="unsloth/GLM-4.6-GGUF"

./llama.cpp/llama-cli \
    --model unsloth/GLM-4.6-GGUF/UD-Q2_K_XL/GLM-4.6-UD-Q2_K_XL-00001-of-00003.gguf \
    --jinja \
    --ctx-size 200000 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 40 \
    --n-gpu-layers 99 \
    -ot ".ffn_.*_exps.=CPU" \
    --cache-type-k q4_1 \
    --cache-type-v q4_1 \
    --flash-attn \
    --threads -1 \
    --in-prefix ' ' \
    --color -i
```

## Python Client Usage

### Interactive Mode
```bash
python3 glm46_client.py --interactive
```

Commands in interactive mode:
- `quit`/`exit` - End conversation
- `clear` - Clear conversation history
- `help` - Show available commands
- `stats` - Show conversation statistics

### Single Query
```bash
python3 glm46_client.py --query "Explain quantum computing" --temperature 0.7
```

### Batch Processing
```bash
# Create prompts.txt with one prompt per line
echo "What is artificial intelligence?" > prompts.txt
echo "Explain the theory of relativity" >> prompts.txt

# Process all prompts
python3 glm46_client.py --batch prompts.txt --output results.json
```

## Performance Tips

### For Better Performance
1. **Use SSD storage** for faster model loading
2. **Increase RAM** if possible (64GB+ recommended for 200K context)
3. **Use GPU with more VRAM** for faster inference
4. **Choose appropriate quantization** based on your hardware

### Memory Usage Optimization
- The script automatically offloads MoE layers to CPU
- KV cache quantization reduces memory usage by ~75%
- Context size can be adjusted if you run into memory issues

### Troubleshooting

#### Out of Memory Errors
1. Try a smaller quantization (e.g., UD-TQ1_0)
2. Reduce context size (modify CONTEXT_SIZE in the script)
3. Close other memory-intensive applications

#### Slow Performance
1. Ensure you're using SSD storage
2. Check if GPU is being utilized (nvidia-smi)
3. Try reducing context size if not needed

#### Server Connection Issues
1. Ensure server is running on correct port (default: 8001)
2. Check firewall settings
3. Verify server URL in client

## Model Information

- **Model**: GLM-4.6 by Z.ai
- **Parameters**: 355B (Mixture of Experts)
- **Context Length**: Up to 200K tokens
- **Special Features**: 
  - State-of-the-art reasoning capabilities
  - Excellent coding performance
  - Improved conversational abilities
  - Tool calling support (experimental)

## Official Settings

The script uses Z.ai's recommended settings:
- Temperature: 1.0
- Top-p: 0.95 (recommended for coding)
- Top-k: 40 (recommended for coding)
- Context: 200K tokens or less

## Support

For issues related to:
- **GLM-4.6 model**: Check [Unsloth documentation](https://docs.unsloth.ai/models/glm-4.6-how-to-run-locally)
- **llama.cpp**: Check [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- **This script**: Create an issue in this repository

## License

This script follows the licenses of the underlying projects:
- llama.cpp: MIT License
- GLM-4.6: Check model license on Hugging Face
- This script: MIT License