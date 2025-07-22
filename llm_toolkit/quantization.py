"""
Production-ready quantization CLI implementation.
Integrates multiple quantization methods with research-based optimizations.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import quantization implementations
try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    GPTQ_AVAILABLE = True
except ImportError:
    GPTQ_AVAILABLE = False

try:
    import awq
    AWQ_AVAILABLE = True
except ImportError:
    AWQ_AVAILABLE = False

from ..advanced_quantization.gptq_implementation import GPTQQuantizer
from ..advanced_quantization.awq_implementation import AWQQuantizer

logger = logging.getLogger(__name__)

class QuantizationCLI:
    """
    Production-ready CLI for model quantization.
    
    Supports multiple quantization methods based on latest research:
    - QLoRA: Efficient fine-tuning with 4-bit quantization
    - GPTQ: GPU-based post-training quantization
    - AWQ: Activation-aware weight quantization
    - SmoothQuant: Smooth activation quantization
    - LLM.int8(): 8-bit inference without degradation
    """
    
    def __init__(self):
        self.supported_methods = {
            'qlora': self._quantize_qlora,
            'gptq': self._quantize_gptq,
            'awq': self._quantize_awq,
            'smoothquant': self._quantize_smoothquant,
            'int8': self._quantize_int8
        }
    
    def run(self, args) -> int:
        """Execute quantization based on CLI arguments."""
        logger.info(f"Starting quantization: {args.method} for model {args.model}")
        
        # Validate method availability
        if not self._check_method_availability(args.method):
            return 1
        
        # Prepare output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            model_name = args.model.split('/')[-1]
            output_dir = Path(f"./quantized_models/{model_name}-{args.method}-{args.bits}bit")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load calibration data if provided
        calibration_data = None
        if args.calibration_data:
            calibration_data = self._load_calibration_data(args.calibration_data)
        
        try:
            # Execute quantization
            quantization_func = self.supported_methods[args.method]
            results = quantization_func(args, output_dir, calibration_data)
            
            # Save results and metadata
            self._save_quantization_metadata(args, results, output_dir)
            
            logger.info(f"Quantization completed successfully!")
            logger.info(f"Model saved to: {output_dir}")
            
            # Print results summary
            self._print_results_summary(results)
            
            return 0
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return 1
    
    def _check_method_availability(self, method: str) -> bool:
        """Check if quantization method dependencies are available."""
        if method == 'gptq' and not GPTQ_AVAILABLE:
            logger.error("GPTQ not available. Install with: pip install auto-gptq")
            return False
        
        if method == 'awq' and not AWQ_AVAILABLE:
            logger.error("AWQ not available. Install with: pip install awq")
            return False
        
        return True
    
    def _load_calibration_data(self, data_path: str) -> List[str]:
        """Load calibration dataset from file."""
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'texts' in data:
                return data['texts']
            else:
                logger.warning("Unexpected calibration data format, using default")
                return None
        
        except Exception as e:
            logger.warning(f"Failed to load calibration data: {e}, using default")
            return None
    
    def _quantize_qlora(self, args, output_dir: Path, calibration_data: Optional[List[str]]) -> Dict[str, Any]:
        """
        QLoRA quantization implementation.
        Based on: Dettmers et al. (2023) - QLoRA: Efficient Finetuning of Quantized LLMs
        """
        logger.info("Applying QLoRA quantization...")
        
        from transformers import BitsAndBytesConfig
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,  # Double quantization for memory efficiency
            bnb_4bit_quant_type="nf4",      # NormalFloat4 for optimal distribution
            bnb_4bit_compute_dtype=torch.bfloat16  # Compute in bfloat16 for stability
        )
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Benchmark the quantized model
        results = self._benchmark_model(model, tokenizer, "QLoRA")
        
        # Save model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        return results
    
    def _quantize_gptq(self, args, output_dir: Path, calibration_data: Optional[List[str]]) -> Dict[str, Any]:
        """
        GPTQ quantization implementation.
        Based on: Frantar et al. (2022) - GPTQ: Accurate Post-Training Quantization
        """
        logger.info("Applying GPTQ quantization...")
        
        quantizer = GPTQQuantizer(
            model_name=args.model,
            bits=args.bits,
            group_size=args.group_size,
            desc_act=False,
            damp_percent=0.1
        )
        
        # Use provided calibration data or default
        if calibration_data is None:
            calibration_data = quantizer._get_default_calibration_data()
        
        # Quantize model
        quantized_model = quantizer.quantize_model(
            calibration_dataset=calibration_data,
            save_dir=str(output_dir)
        )
        
        # Benchmark results
        results = quantizer.benchmark_model(quantized_model)
        results['method'] = 'GPTQ'
        
        return results
    
    def _quantize_awq(self, args, output_dir: Path, calibration_data: Optional[List[str]]) -> Dict[str, Any]:
        """
        AWQ quantization implementation.
        Based on: Lin et al. (2023) - AWQ: Activation-aware Weight Quantization
        """
        logger.info("Applying AWQ quantization...")
        
        quantizer = AWQQuantizer(
            model_name=args.model,
            w_bit=args.bits,
            q_group_size=args.group_size,
            zero_point=True
        )
        
        # Default calibration data if not provided
        if calibration_data is None:
            calibration_data = [
                "The quick brown fox jumps over the lazy dog.",
                "Artificial intelligence is transforming various industries.",
                "Climate change requires immediate global action.",
                "The human genome contains billions of base pairs.",
                "Quantum computers leverage quantum mechanical phenomena."
            ]
        
        # Apply quantization
        quantized_model = quantizer.apply_awq_quantization(calibration_data)
        
        # Save model
        quantized_model.save_pretrained(output_dir)
        quantizer.tokenizer.save_pretrained(output_dir)
        
        # Benchmark results
        results = quantizer.benchmark_awq_model()
        results['method'] = 'AWQ'
        
        return results
    
    def _quantize_smoothquant(self, args, output_dir: Path, calibration_data: Optional[List[str]]) -> Dict[str, Any]:
        """
        SmoothQuant implementation.
        Based on: Xiao et al. (2022) - SmoothQuant: Accurate and Efficient Post-Training Quantization
        """
        logger.info("Applying SmoothQuant quantization...")
        
        # Load model in FP16 first
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Apply SmoothQuant transformations
        # This is a simplified implementation - full SmoothQuant requires more complex transformations
        logger.warning("SmoothQuant implementation is simplified. For full implementation, use official SmoothQuant library.")
        
        # Convert to int8 using PyTorch's built-in quantization
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        # Benchmark
        results = self._benchmark_model(model, tokenizer, "SmoothQuant")
        
        # Save model
        torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
        tokenizer.save_pretrained(output_dir)
        
        return results
    
    def _quantize_int8(self, args, output_dir: Path, calibration_data: Optional[List[str]]) -> Dict[str, Any]:
        """
        LLM.int8() quantization implementation.
        Based on: Dettmers et al. (2022) - LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
        """
        logger.info("Applying LLM.int8() quantization...")
        
        from transformers import BitsAndBytesConfig
        
        # Configure 8-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,  # Threshold for outlier detection
            llm_int8_has_fp16_weight=False
        )
        
        # Load model with 8-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Benchmark
        results = self._benchmark_model(model, tokenizer, "LLM.int8()")
        
        # Save model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        return results
    
    def _benchmark_model(self, model, tokenizer, method_name: str) -> Dict[str, Any]:
        """Benchmark quantized model performance."""
        logger.info(f"Benchmarking {method_name} model...")
        
        # Calculate model size
        model_size = 0
        for param in model.parameters():
            model_size += param.numel() * param.element_size()
        model_size_mb = model_size / (1024 * 1024)
        
        # Inference speed test
        test_input = "The future of artificial intelligence is"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model.generate(**inputs, max_length=50, do_sample=False)
        
        # Timing
        import time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                outputs = model.generate(**inputs, max_length=50, do_sample=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        avg_inference_time = (end_time - start_time) / 10
        
        return {
            "method": method_name,
            "model_size_mb": model_size_mb,
            "avg_inference_time_seconds": avg_inference_time,
            "tokens_per_second": 50 / avg_inference_time,  # Approximate
        }
    
    def _save_quantization_metadata(self, args, results: Dict[str, Any], output_dir: Path):
        """Save quantization metadata and results."""
        metadata = {
            "quantization_config": {
                "method": args.method,
                "bits": args.bits,
                "group_size": getattr(args, 'group_size', None),
                "model": args.model,
            },
            "results": results,
            "toolkit_version": "1.0.0"
        }
        
        with open(output_dir / "quantization_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _print_results_summary(self, results: Dict[str, Any]):
        """Print quantization results summary."""
        print("\n" + "="*50)
        print(f"QUANTIZATION RESULTS - {results['method']}")
        print("="*50)
        print(f"Model Size: {results['model_size_mb']:.2f} MB")
        print(f"Inference Time: {results['avg_inference_time_seconds']:.4f} seconds")
        print(f"Tokens/Second: {results.get('tokens_per_second', 'N/A')}")
        
        if 'compression_ratio' in results:
            print(f"Compression Ratio: {results['compression_ratio']}")
        if 'memory_reduction' in results:
            print(f"Memory Reduction: {results['memory_reduction']}")
        
        print("="*50)