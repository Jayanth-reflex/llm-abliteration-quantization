# 📚 API Reference

Welcome to the LLM Optimization Toolkit API documentation. This comprehensive reference covers all modules, classes, and functions available in the toolkit.

## 🏗️ Architecture Overview

The toolkit is organized into several key modules:

```
llm_toolkit/
├── quantization.py      # Core quantization functionality
├── abliteration.py      # Model abliteration methods
├── multimodal.py        # Multi-modal optimization
├── distributed.py       # Distributed quantization
└── __main__.py          # CLI entry point

advanced_quantization/
├── gptq_implementation.py    # GPTQ algorithm
├── awq_implementation.py     # AWQ algorithm
└── smoothquant.py           # SmoothQuant method

research_extensions/
└── combined_optimization.py  # Novel research methods
```

## 🔧 Core Modules

### llm_toolkit.quantization

The main quantization module providing production-ready quantization methods.

#### Classes

##### `QuantizationCLI`

Main CLI interface for quantization operations.

```python
class QuantizationCLI:
    """
    Production-ready CLI for model quantization.
    
    Supports multiple quantization methods:
    - QLoRA: Efficient fine-tuning with 4-bit quantization
    - GPTQ: GPU-based post-training quantization
    - AWQ: Activation-aware weight quantization
    - SmoothQuant: Smooth activation quantization
    - LLM.int8(): 8-bit inference without degradation
    """
    
    def __init__(self):
        """Initialize quantization CLI."""
        
    def run(self, args) -> int:
        """
        Execute quantization based on CLI arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Exit code (0 for success, 1 for failure)
        """
```

#### Functions

##### `quantize_model()`

```python
def quantize_model(
    model_name: str,
    method: str = "qlora",
    bits: int = 4,
    output_dir: Optional[str] = None,
    calibration_data: Optional[List[str]] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Quantize a language model using specified method.
    
    Args:
        model_name: HuggingFace model identifier
        method: Quantization method ('qlora', 'gptq', 'awq', 'smoothquant', 'int8')
        bits: Number of quantization bits (2, 3, 4, 8)
        output_dir: Directory to save quantized model
        calibration_data: Optional calibration dataset
        
    Returns:
        Tuple of (quantized_model, benchmark_results)
        
    Raises:
        ValueError: If method or bits not supported
        RuntimeError: If quantization fails
        
    Example:
        >>> model, results = quantize_model("gpt2", "qlora", bits=4)
        >>> print(f"Memory reduction: {results['compression_ratio']:.1f}x")
    """
```

### llm_toolkit.abliteration

Model abliteration and behavior modification methods.

#### Classes

##### `AbliterationCLI`

```python
class AbliterationCLI:
    """
    Production-ready CLI for model abliteration.
    
    Supports multiple abliteration methods:
    - Inference-time intervention
    - Weight orthogonalization  
    - Selective abliteration for specific topics
    - Combined approaches
    """
    
    def __init__(self):
        """Initialize abliteration CLI."""
        
    def run(self, args) -> int:
        """Execute abliteration based on CLI arguments."""
```

#### Functions

##### `abliterate_model()`

```python
def abliterate_model(
    model_name: str,
    method: str = "inference",
    strength: float = 0.8,
    target_topics: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Remove refusal behaviors from a language model.
    
    Args:
        model_name: HuggingFace model identifier
        method: Abliteration method ('inference', 'orthogonal', 'selective', 'combined')
        strength: Abliteration strength (0.0-1.0)
        target_topics: Specific topics for selective abliteration
        output_dir: Directory to save abliterated model
        
    Returns:
        Tuple of (abliterated_model, effectiveness_metrics)
        
    Example:
        >>> model, metrics = abliterate_model("gpt2", "selective", 
        ...                                   target_topics=["violence", "illegal"])
        >>> print(f"Effectiveness: {metrics['effectiveness']:.2f}")
    """
```

### llm_toolkit.multimodal

Multi-modal model optimization for vision-language models.

#### Classes

##### `MultiModalOptimizer`

```python
class MultiModalOptimizer:
    """
    Multi-modal model optimization for vision-language models.
    
    Supports:
    - CLIP (Contrastive Language-Image Pre-training)
    - BLIP-2 (Bootstrapped Vision-Language Pre-training)
    - LLaVA (Large Language and Vision Assistant)
    - Custom vision-language architectures
    """
    
    def optimize_model(
        self,
        model_name: str,
        optimize: str = "both",
        vision_bits: int = 8,
        language_bits: int = 4
    ) -> Dict[str, Any]:
        """
        Optimize multi-modal model components.
        
        Args:
            model_name: Multi-modal model identifier
            optimize: Components to optimize ('vision', 'language', 'both')
            vision_bits: Vision encoder quantization bits
            language_bits: Language model quantization bits
            
        Returns:
            Optimization results and metrics
        """
```

### llm_toolkit.distributed

Distributed quantization across multiple GPUs.

#### Classes

##### `DistributedQuantizer`

```python
class DistributedQuantizer:
    """
    Distributed quantization for large models across multiple GPUs.
    
    Strategies:
    - Tensor Parallel: Split tensors across GPUs
    - Pipeline Parallel: Split layers across GPUs
    - Hybrid: Combination of tensor and pipeline parallelism
    - Data Parallel: Replicate model, split data
    """
    
    def quantize_distributed(
        self,
        model_name: str,
        strategy: str = "tensor_parallel",
        num_gpus: int = 2,
        bits: int = 4
    ) -> Dict[str, Any]:
        """
        Quantize model using distributed strategy.
        
        Args:
            model_name: Model to quantize
            strategy: Distribution strategy
            num_gpus: Number of GPUs to use
            bits: Quantization bits
            
        Returns:
            Distributed quantization results
        """
```

## 🔬 Advanced Quantization

### advanced_quantization.gptq_implementation

GPTQ (Accurate Post-Training Quantization) implementation.

#### Classes

##### `GPTQQuantizer`

```python
class GPTQQuantizer:
    """
    GPTQ quantization implementation following the original paper.
    
    Key innovations from the paper:
    1. Layer-wise quantization to minimize error propagation
    2. Optimal Brain Surgeon (OBS) based weight updates
    3. Efficient GPU implementation for large models
    """
    
    def __init__(
        self,
        model_name: str,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
        damp_percent: float = 0.1
    ):
        """
        Initialize GPTQ quantizer.
        
        Args:
            model_name: HuggingFace model identifier
            bits: Quantization bits (2, 3, 4, 8)
            group_size: Group size for quantization (-1 for no grouping)
            desc_act: Whether to use descending activation order
            damp_percent: Damping factor for numerical stability
        """
    
    def quantize_model(
        self, 
        calibration_dataset: Optional[list] = None,
        save_dir: Optional[str] = None
    ) -> AutoGPTQForCausalLM:
        """
        Quantize model using GPTQ algorithm.
        
        Research insights implemented:
        - Uses calibration data for optimal quantization points
        - Applies layer-wise quantization to minimize error accumulation
        - Implements efficient GPU kernels for inference
        """
```

### advanced_quantization.awq_implementation

AWQ (Activation-aware Weight Quantization) implementation.

#### Classes

##### `AWQQuantizer`

```python
class AWQQuantizer:
    """
    AWQ (Activation-aware Weight Quantization) implementation.
    
    Key research contributions:
    1. Activation-aware weight importance scoring
    2. Per-channel scaling for optimal quantization
    3. Hardware-efficient 4-bit inference
    """
    
    def __init__(
        self,
        model_name: str,
        w_bit: int = 4,
        q_group_size: int = 128,
        zero_point: bool = True,
        version: str = "GEMM"
    ):
        """Initialize AWQ quantizer."""
    
    def apply_awq_quantization(self, calibration_data: List[str]) -> nn.Module:
        """Apply AWQ quantization to the entire model."""
```

## 🧪 Research Extensions

### research_extensions.combined_optimization

Novel research combining abliteration and quantization.

#### Classes

##### `CombinedOptimizer`

```python
class CombinedOptimizer:
    """
    Research implementation combining abliteration and quantization techniques.
    
    Key Research Contributions:
    1. Quantization-aware abliteration: Modify refusal directions post-quantization
    2. Selective abliteration: Target specific topics while preserving others
    3. Efficiency analysis: Compare different combination strategies
    """
    
    def __init__(
        self,
        model_name: str,
        quantization_config: Optional[Dict] = None,
        abliteration_config: Optional[Dict] = None
    ):
        """Initialize combined optimizer."""
    
    def apply_combined_approach(self) -> nn.Module:
        """Apply combined quantization and abliteration."""
```

## 📊 Benchmarking

### benchmarks.comprehensive_benchmark

Research-grade benchmarking suite.

#### Classes

##### `ComprehensiveBenchmark`

```python
class ComprehensiveBenchmark:
    """
    Research-grade benchmarking suite for LLM optimization methods.
    
    Implements evaluation protocols from major research papers:
    - QLoRA (Dettmers et al., 2023)
    - GPTQ (Frantar et al., 2022)
    - AWQ (Lin et al., 2023)
    """
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark suite with configuration."""
    
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """Run comprehensive benchmark across all configurations."""
```

#### Data Classes

##### `BenchmarkConfig`

```python
@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    model_names: List[str]
    quantization_methods: List[str]
    bits: List[int]
    batch_sizes: List[int]
    sequence_lengths: List[int]
    num_runs: int = 3
    warmup_runs: int = 2
    output_dir: str = "./benchmark_results"
    save_plots: bool = True
    detailed_analysis: bool = True
```

##### `BenchmarkResult`

```python
@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    method: str
    bits: int
    memory_usage_mb: float
    inference_time_ms: float
    throughput_tokens_per_sec: float
    model_size_mb: float
    perplexity: Optional[float] = None
    accuracy_scores: Optional[Dict[str, float]] = None
    hardware_utilization: Optional[Dict[str, float]] = None
```

## 🎯 Usage Examples

### Basic Quantization

```python
from llm_toolkit.quantization import QuantizationCLI

# Initialize CLI
cli = QuantizationCLI()

# Create mock arguments (normally from argparse)
class Args:
    model = "gpt2"
    method = "qlora"
    bits = 4
    output = "./quantized_model"
    calibration_data = None
    group_size = 128

# Run quantization
result = cli.run(Args())
```

### Advanced GPTQ Usage

```python
from advanced_quantization.gptq_implementation import GPTQQuantizer

# Initialize quantizer
quantizer = GPTQQuantizer(
    model_name="facebook/opt-125m",
    bits=4,
    group_size=128
)

# Quantize with custom calibration data
calibration_data = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming the world.",
    "Quantum computing represents the future."
]

quantized_model = quantizer.quantize_model(
    calibration_dataset=calibration_data,
    save_dir="./gptq_quantized"
)

# Benchmark results
results = quantizer.benchmark_model(quantized_model)
print(f"Compression ratio: {results['compression_ratio']}")
```

### Combined Optimization

```python
from research_extensions.combined_optimization import CombinedOptimizer

# Initialize optimizer
optimizer = CombinedOptimizer(
    model_name="microsoft/DialoGPT-small",
    quantization_config={
        "method": "qlora",
        "bits": 4
    },
    abliteration_config={
        "method": "selective",
        "strength": 0.8,
        "target_topics": ["violence", "illegal"]
    }
)

# Apply combined optimization
optimized_model = optimizer.apply_combined_approach()

# Benchmark effectiveness
results = optimizer.benchmark_combined_approach({
    "harmful": ["How to make weapons"],
    "normal": ["Explain quantum physics"]
})
```

### Comprehensive Benchmarking

```python
from benchmarks.comprehensive_benchmark import ComprehensiveBenchmark, BenchmarkConfig

# Configure benchmark
config = BenchmarkConfig(
    model_names=["gpt2", "microsoft/DialoGPT-small"],
    quantization_methods=["baseline", "qlora", "gptq"],
    bits=[4, 8],
    batch_sizes=[1, 4],
    sequence_lengths=[128, 512],
    num_runs=3
)

# Run benchmark
benchmark = ComprehensiveBenchmark(config)
results_df = benchmark.run_comprehensive_benchmark()

# Analyze results
print(results_df.groupby(["Method", "Bits"]).agg({
    "Memory (MB)": "mean",
    "Throughput (tokens/s)": "mean"
}))
```

## 🔧 Configuration

### Environment Variables

```bash
# Hugging Face cache directory
export TRANSFORMERS_CACHE=/path/to/cache

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

# CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Configuration Files

The toolkit supports configuration files in JSON format:

```json
{
  "quantization": {
    "default_method": "qlora",
    "default_bits": 4,
    "cache_dir": "./model_cache"
  },
  "abliteration": {
    "default_strength": 0.8,
    "default_method": "inference"
  },
  "benchmarking": {
    "num_runs": 3,
    "warmup_runs": 2,
    "save_plots": true
  }
}
```

## 🐛 Error Handling

### Common Exceptions

#### `QuantizationError`

```python
class QuantizationError(Exception):
    """Raised when quantization fails."""
    pass
```

#### `AbliterationError`

```python
class AbliterationError(Exception):
    """Raised when abliteration fails."""
    pass
```

#### `BenchmarkError`

```python
class BenchmarkError(Exception):
    """Raised when benchmarking fails."""
    pass
```

### Error Handling Examples

```python
from llm_toolkit.quantization import quantize_model, QuantizationError

try:
    model, results = quantize_model("gpt2", "qlora", bits=4)
except QuantizationError as e:
    print(f"Quantization failed: {e}")
    # Handle error appropriately
except ValueError as e:
    print(f"Invalid parameters: {e}")
    # Handle parameter errors
```

## 📈 Performance Considerations

### Memory Management

```python
import torch

# Clear CUDA cache after operations
torch.cuda.empty_cache()

# Use gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Monitor memory usage
if torch.cuda.is_available():
    memory_used = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak memory usage: {memory_used:.1f} MB")
```

### Optimization Tips

1. **Batch Size**: Start with batch_size=1 for large models
2. **Sequence Length**: Use shorter sequences for initial testing
3. **Device Mapping**: Use `device_map="auto"` for automatic GPU distribution
4. **Data Types**: Use `torch.float16` or `torch.bfloat16` for efficiency
5. **Compilation**: Use `torch.compile()` for PyTorch 2.0+ speedups

## 🔗 Integration Examples

### Hugging Face Integration

```python
from transformers import pipeline
from llm_toolkit.quantization import quantize_model

# Quantize model
model, _ = quantize_model("gpt2", "qlora", bits=4)

# Create pipeline with quantized model
generator = pipeline("text-generation", model=model, tokenizer="gpt2")

# Generate text
output = generator("The future of AI is", max_length=50)
print(output[0]['generated_text'])
```

### FastAPI Integration

```python
from fastapi import FastAPI
from llm_toolkit.quantization import quantize_model

app = FastAPI()

# Load quantized model at startup
model, tokenizer = None, None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model, _ = quantize_model("gpt2", "qlora", bits=4)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

@app.post("/generate")
async def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return {"generated": tokenizer.decode(outputs[0])}
```

---

## 📞 Support

For API-related questions:
- **GitHub Issues**: [Report bugs or request features](https://github.com/your-repo/issues)
- **Discussions**: [Community Q&A](https://github.com/your-repo/discussions)
- **Documentation**: [Full documentation](https://llm-optimization.readthedocs.io)

---

**This API reference is automatically updated with each release. For the latest information, always refer to the online documentation.**