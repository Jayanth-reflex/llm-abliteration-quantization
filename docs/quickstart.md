# 🚀 Quick Start Guide

Welcome to the LLM Optimization Toolkit! This guide will get you up and running in minutes, whether you're a beginner or an advanced practitioner.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8+ (3.10+ recommended)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space

### Hardware Recommendations
| Use Case | GPU | RAM | Storage |
|----------|-----|-----|---------|
| Learning/Small Models | GTX 1080 (8GB) | 16GB | 20GB |
| Research/Medium Models | RTX 3080 (12GB) | 32GB | 50GB |
| Production/Large Models | RTX 4090 (24GB) | 64GB | 100GB |

## 🔧 Installation

### Option 1: Quick Install (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-repo/llm-optimization
cd llm-optimization

# Install with conda (recommended)
conda create -n llm-opt python=3.10
conda activate llm-opt
pip install -r requirements.txt

# Verify installation
python -m llm_toolkit --version
```

### Option 2: Development Install
```bash
# For contributors and advanced users
git clone https://github.com/your-repo/llm-optimization
cd llm-optimization

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

### Option 3: Docker Install
```bash
# Pull pre-built image
docker pull llm-optimization:latest

# Or build locally
docker build -t llm-optimization .
docker run -it --gpus all llm-optimization
```

## 🎯 Your First 5 Minutes

### 1. Test Your Installation (30 seconds)
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Quantize Your First Model (2 minutes)
```bash
# Start with a small model
python -m llm_toolkit quantize \
    --model facebook/opt-125m \
    --method qlora \
    --bits 4 \
    --output ./my_first_quantized_model
```

### 3. Test the Quantized Model (1 minute)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your quantized model
model = AutoModelForCausalLM.from_pretrained("./my_first_quantized_model")
tokenizer = AutoTokenizer.from_pretrained("./my_first_quantized_model")

# Generate text
inputs = tokenizer("The future of AI is", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 4. Compare Methods (1.5 minutes)
```bash
# Try different quantization methods
python -m llm_toolkit quantize --model facebook/opt-125m --method gptq --bits 4
python -m llm_toolkit quantize --model facebook/opt-125m --method awq --bits 4

# Compare results
python scripts/compare_models.py --models ./opt-125m-* --benchmark speed memory quality
```

## 🎓 Learning Paths

### Path 1: Complete Beginner (2-4 hours)
```bash
# Start here if you're new to quantization
cd tutorials/beginner/

# Interactive tutorial
jupyter notebook 01_quantization_basics.ipynb
jupyter notebook 02_your_first_quantization.ipynb
jupyter notebook 03_understanding_results.ipynb
```

### Path 2: Intermediate Developer (4-8 hours)
```bash
# For developers with ML experience
cd tutorials/intermediate/

# Advanced techniques
jupyter notebook 01_advanced_quantization.ipynb
jupyter notebook 02_abliteration_basics.ipynb
jupyter notebook 03_multimodal_optimization.ipynb
jupyter notebook 04_custom_implementations.ipynb
```

### Path 3: Research/Advanced (8+ hours)
```bash
# For researchers and advanced practitioners
cd tutorials/advanced/

# Research-level content
jupyter notebook 01_paper_implementations.ipynb
jupyter notebook 02_novel_techniques.ipynb
jupyter notebook 03_benchmarking_suite.ipynb
jupyter notebook 04_contributing_research.ipynb
```

## 🔥 Common Use Cases

### Use Case 1: Reduce Model Size for Deployment
```bash
# Quantize a 7B model to fit on consumer GPU
python -m llm_toolkit quantize \
    --model meta-llama/Llama-2-7b-hf \
    --method qlora \
    --bits 4 \
    --output ./llama2-7b-4bit

# Expected: 13GB → 3.5GB (4x compression)
```

### Use Case 2: Remove Model Refusals for Research
```bash
# Abliterate safety filters for research purposes
python -m llm_toolkit abliterate \
    --model microsoft/DialoGPT-medium \
    --method selective \
    --strength 0.8 \
    --target-topics harmful illegal \
    --output ./dialogpt-abliterated
```

### Use Case 3: Optimize Multimodal Models
```bash
# Optimize CLIP for faster inference
python -m llm_toolkit multimodal \
    --model openai/clip-vit-base-patch32 \
    --optimize both \
    --vision-bits 8 \
    --language-bits 4 \
    --output ./clip-optimized
```

### Use Case 4: Distributed Training Setup
```bash
# Distribute large model across multiple GPUs
python -m llm_toolkit distributed \
    --model meta-llama/Llama-2-13b-hf \
    --gpus 4 \
    --strategy tensor_parallel \
    --bits 4 \
    --output ./llama2-13b-distributed
```

## 🛠️ Essential Commands

### Model Information
```bash
# Get model info
python -m llm_toolkit info --model facebook/opt-350m

# Compare models
python -m llm_toolkit compare --models model1 model2 model3

# Benchmark model
python -m llm_toolkit benchmark --model ./my_model --tasks speed memory quality
```

### Batch Processing
```bash
# Process multiple models
python -m llm_toolkit batch-quantize \
    --models facebook/opt-125m facebook/opt-350m \
    --method qlora \
    --bits 4

# Batch abliteration
python -m llm_toolkit batch-abliterate \
    --models gpt2 gpt2-medium \
    --strength 0.8
```

### Advanced Options
```bash
# Custom calibration data
python -m llm_toolkit quantize \
    --model facebook/opt-350m \
    --method gptq \
    --calibration-data ./my_calibration.json \
    --group-size 64

# Selective abliteration with custom topics
python -m llm_toolkit abliterate \
    --model gpt2 \
    --method selective \
    --refusal-data ./custom_refusals.json \
    --target-topics violence drugs weapons
```

## 🔍 Troubleshooting

### Common Issues

#### Issue 1: CUDA Out of Memory
```bash
# Solution: Use smaller batch size or model sharding
python -m llm_toolkit quantize \
    --model facebook/opt-350m \
    --method qlora \
    --batch-size 1 \
    --device-map auto
```

#### Issue 2: Import Errors
```bash
# Check dependencies
pip install --upgrade transformers torch bitsandbytes

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

#### Issue 3: Slow Performance
```bash
# Enable optimizations
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=true

# Use faster methods
python -m llm_toolkit quantize --method awq  # Usually faster than GPTQ
```

### Getting Help
```bash
# Built-in help
python -m llm_toolkit --help
python -m llm_toolkit quantize --help

# Verbose output for debugging
python -m llm_toolkit quantize --model facebook/opt-125m --verbose

# Check system compatibility
python scripts/check_system.py
```

## 📊 Performance Expectations

### Quantization Results (Typical)
| Model Size | Original | 4-bit | 8-bit | Speed | Quality |
|------------|----------|-------|-------|-------|---------|
| 125M | 500MB | 125MB | 250MB | 2x | 98% |
| 350M | 1.4GB | 350MB | 700MB | 2.5x | 97% |
| 1.3B | 5.2GB | 1.3GB | 2.6GB | 3x | 96% |
| 7B | 28GB | 7GB | 14GB | 4x | 95% |

### Hardware Performance
| GPU | Model Size | Method | Time | Memory |
|-----|------------|--------|------|--------|
| RTX 3080 | 7B | QLoRA | 15min | 12GB |
| RTX 4090 | 13B | GPTQ | 25min | 20GB |
| A100 | 30B | AWQ | 45min | 40GB |

## 🎯 Next Steps

### For Beginners
1. Complete the [Beginner Tutorial Series](../tutorials/beginner/)
2. Try the [Interactive Examples](../examples/interactive/)
3. Join our [Community Discord](https://discord.gg/llm-optimization)

### For Developers
1. Explore [Advanced Techniques](../advanced_quantization/)
2. Read [Paper Implementations](../educational_content/paper_implementations/)
3. Contribute to [Open Issues](https://github.com/your-repo/issues)

### For Researchers
1. Review [Research Extensions](../research_extensions/)
2. Implement [Novel Techniques](../tutorials/advanced/novel_techniques.ipynb)
3. Submit [Research Contributions](../CONTRIBUTING.md#research-contributions)

## 📚 Additional Resources

- **Documentation**: [Full Documentation](./index.md)
- **API Reference**: [API Docs](./api/)
- **Examples**: [Code Examples](../examples/)
- **Tutorials**: [Tutorial Index](../tutorials/)
- **Community**: [Discussions](https://github.com/your-repo/discussions)

---

**Ready to dive deeper? Choose your learning path above or explore our [comprehensive tutorials](../tutorials/)!**