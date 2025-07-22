#!/usr/bin/env python3
"""
LLM Model Analyzer - Comprehensive Model Analysis Tool

This tool provides deep analysis of language models including:
- Architecture analysis and visualization
- Performance profiling and bottleneck detection
- Memory usage breakdown and optimization suggestions
- Quantization impact assessment
- Hardware compatibility analysis

Usage:
    python tools/model_analyzer.py --model gpt2 --analysis all
    python tools/model_analyzer.py --model llama2-7b --analysis memory --quantization 4bit
    python tools/model_analyzer.py --compare gpt2,llama2-7b --metric speed

Author: LLM Optimization Toolkit
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import psutil
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ModelAnalyzer:
    """Comprehensive model analysis and profiling tool."""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.config = None
        self.analysis_results = {}
        
    def load_model(self, quantization: Optional[str] = None):
        """Load model with optional quantization."""
        print(f"🔍 Loading model: {self.model_name}")
        
        try:
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if quantization:
                from transformers import BitsAndBytesConfig
                
                if quantization == "4bit":
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                elif quantization == "8bit":
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    quant_config = None
                
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    quantization_config=quant_config,
                    device_map=self.device
                )
            else:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    device_map=self.device
                )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def analyze_architecture(self) -> Dict[str, Any]:
        """Analyze model architecture and structure."""
        print("🏗️ Analyzing model architecture...")
        
        architecture = {
            "model_type": self.config.model_type,
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "layers": {},
            "attention_heads": getattr(self.config, 'num_attention_heads', 'N/A'),
            "hidden_size": getattr(self.config, 'hidden_size', 'N/A'),
            "vocab_size": getattr(self.config, 'vocab_size', 'N/A'),
            "max_position_embeddings": getattr(self.config, 'max_position_embeddings', 'N/A')
        }
        
        # Analyze layer distribution
        layer_counts = {}
        param_distribution = {}
        
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            layer_counts[module_type] = layer_counts.get(module_type, 0) + 1
            
            if hasattr(module, 'weight') and module.weight is not None:
                param_count = module.weight.numel()
                param_distribution[name] = param_count
        
        architecture["layer_counts"] = layer_counts
        architecture["parameter_distribution"] = param_distribution
        
        # Calculate model size in different precisions
        total_params = architecture["total_parameters"]
        architecture["model_sizes"] = {
            "fp32": f"{total_params * 4 / 1e9:.2f} GB",
            "fp16": f"{total_params * 2 / 1e9:.2f} GB",
            "int8": f"{total_params * 1 / 1e9:.2f} GB",
            "int4": f"{total_params * 0.5 / 1e9:.2f} GB"
        }
        
        self.analysis_results["architecture"] = architecture
        return architecture
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        print("💾 Analyzing memory usage...")
        
        memory_analysis = {}
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Measure memory for different operations
            test_input = torch.randint(0, 1000, (1, 512)).to(self.model.device)
            
            # Forward pass memory
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = self.model(test_input)
            forward_memory = torch.cuda.max_memory_allocated() / 1e9
            
            # Memory by layer
            layer_memory = {}
            hooks = []
            
            def memory_hook(name):
                def hook(module, input, output):
                    if torch.cuda.is_available():
                        layer_memory[name] = torch.cuda.memory_allocated() / 1e9
                return hook
            
            # Register hooks
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Embedding)):
                    hooks.append(module.register_forward_hook(memory_hook(name)))
            
            # Run forward pass with hooks
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = self.model(test_input)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            memory_analysis = {
                "forward_pass_memory_gb": forward_memory,
                "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                "layer_memory_gb": layer_memory,
                "gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            }
        else:
            # CPU memory estimation
            model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e9
            memory_analysis = {
                "model_size_gb": model_size,
                "estimated_inference_memory_gb": model_size * 1.5,  # Rough estimate
                "system_memory_gb": psutil.virtual_memory().total / 1e9,
                "available_memory_gb": psutil.virtual_memory().available / 1e9
            }
        
        self.analysis_results["memory"] = memory_analysis
        return memory_analysis
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze inference performance."""
        print("⚡ Analyzing performance...")
        
        # Test prompts of varying lengths
        test_prompts = [
            "Hello",
            "The quick brown fox jumps over the lazy dog",
            "In the field of artificial intelligence, machine learning algorithms have revolutionized",
            "The rapid advancement of technology in the 21st century has fundamentally transformed how we communicate, work, and interact with the world around us, creating unprecedented opportunities"
        ]
        
        performance_results = {
            "latency_by_length": {},
            "throughput_analysis": {},
            "batch_performance": {}
        }
        
        # Latency analysis by input length
        for prompt in test_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_length = inputs['input_ids'].shape[1]
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = self.model(**inputs)
            
            # Timing
            times = []
            for _ in range(10):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            performance_results["latency_by_length"][input_length] = {
                "avg_latency_ms": np.mean(times),
                "std_latency_ms": np.std(times),
                "min_latency_ms": np.min(times),
                "max_latency_ms": np.max(times)
            }
        
        # Batch performance analysis
        batch_sizes = [1, 2, 4, 8] if torch.cuda.is_available() else [1, 2]
        
        for batch_size in batch_sizes:
            try:
                # Create batch input
                batch_input = self.tokenizer(
                    ["Test prompt"] * batch_size,
                    return_tensors="pt",
                    padding=True
                ).to(self.model.device)
                
                # Warmup
                for _ in range(3):
                    with torch.no_grad():
                        _ = self.model(**batch_input)
                
                # Timing
                times = []
                for _ in range(5):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    start_time = time.time()
                    with torch.no_grad():
                        _ = self.model(**batch_input)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                throughput = batch_size / avg_time
                
                performance_results["batch_performance"][batch_size] = {
                    "avg_time_seconds": avg_time,
                    "throughput_samples_per_second": throughput,
                    "time_per_sample_ms": (avg_time / batch_size) * 1000
                }
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ Batch size {batch_size} caused OOM")
                    break
                else:
                    raise
        
        self.analysis_results["performance"] = performance_results
        return performance_results
    
    def analyze_quantization_impact(self, original_model_name: str) -> Dict[str, Any]:
        """Compare quantized model with original."""
        print("🔬 Analyzing quantization impact...")
        
        # This would require loading both models and comparing
        # For now, provide theoretical analysis
        quantization_analysis = {
            "theoretical_speedup": "2-4x faster inference",
            "theoretical_memory_reduction": "4x smaller for 4-bit, 2x for 8-bit",
            "expected_quality_retention": "95-98% for most tasks",
            "hardware_compatibility": "Broader deployment options"
        }
        
        self.analysis_results["quantization_impact"] = quantization_analysis
        return quantization_analysis
    
    def generate_visualizations(self, output_dir: str = "./analysis_results"):
        """Generate analysis visualizations."""
        print("📊 Generating visualizations...")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Architecture visualization
        if "architecture" in self.analysis_results:
            arch = self.analysis_results["architecture"]
            
            # Parameter distribution pie chart
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            layer_counts = arch["layer_counts"]
            plt.pie(layer_counts.values(), labels=layer_counts.keys(), autopct='%1.1f%%')
            plt.title("Layer Type Distribution")
            
            plt.subplot(2, 2, 2)
            sizes = [float(size.split()[0]) for size in arch["model_sizes"].values()]
            precisions = list(arch["model_sizes"].keys())
            plt.bar(precisions, sizes)
            plt.title("Model Size by Precision")
            plt.ylabel("Size (GB)")
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/architecture_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Performance visualization
        if "performance" in self.analysis_results:
            perf = self.analysis_results["performance"]
            
            plt.figure(figsize=(12, 6))
            
            # Latency by input length
            if "latency_by_length" in perf:
                plt.subplot(1, 2, 1)
                lengths = list(perf["latency_by_length"].keys())
                latencies = [perf["latency_by_length"][l]["avg_latency_ms"] for l in lengths]
                errors = [perf["latency_by_length"][l]["std_latency_ms"] for l in lengths]
                
                plt.errorbar(lengths, latencies, yerr=errors, marker='o')
                plt.xlabel("Input Length (tokens)")
                plt.ylabel("Latency (ms)")
                plt.title("Latency vs Input Length")
                plt.grid(True, alpha=0.3)
            
            # Batch performance
            if "batch_performance" in perf:
                plt.subplot(1, 2, 2)
                batch_sizes = list(perf["batch_performance"].keys())
                throughputs = [perf["batch_performance"][b]["throughput_samples_per_second"] for b in batch_sizes]
                
                plt.plot(batch_sizes, throughputs, marker='s')
                plt.xlabel("Batch Size")
                plt.ylabel("Throughput (samples/sec)")
                plt.title("Throughput vs Batch Size")
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/performance_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Visualizations saved to {output_dir}")
    
    def generate_report(self, output_file: str = "model_analysis_report.json"):
        """Generate comprehensive analysis report."""
        print("📋 Generating analysis report...")
        
        report = {
            "model_name": self.model_name,
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_results": self.analysis_results,
            "recommendations": self._generate_recommendations()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Report saved to {output_file}")
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        if "architecture" in self.analysis_results:
            arch = self.analysis_results["architecture"]
            total_params = arch["total_parameters"]
            
            if total_params > 1e9:  # > 1B parameters
                recommendations.append("Consider 4-bit quantization for significant memory savings")
                recommendations.append("Use gradient checkpointing for training to reduce memory")
                recommendations.append("Consider model parallelism for large-scale deployment")
            
            if total_params > 10e9:  # > 10B parameters
                recommendations.append("Implement distributed inference across multiple GPUs")
                recommendations.append("Consider using specialized hardware (A100, H100)")
        
        if "memory" in self.analysis_results:
            memory = self.analysis_results["memory"]
            
            if "forward_pass_memory_gb" in memory and memory["forward_pass_memory_gb"] > 8:
                recommendations.append("High memory usage detected - consider quantization")
                recommendations.append("Use mixed precision training (FP16/BF16)")
        
        if "performance" in self.analysis_results:
            perf = self.analysis_results["performance"]
            
            if "batch_performance" in perf:
                best_batch = max(perf["batch_performance"].items(), 
                                key=lambda x: x[1]["throughput_samples_per_second"])
                recommendations.append(f"Optimal batch size appears to be {best_batch[0]} for throughput")
        
        return recommendations

def compare_models(model_names: List[str], metric: str = "all") -> Dict[str, Any]:
    """Compare multiple models on specified metrics."""
    print(f"🔄 Comparing models: {', '.join(model_names)}")
    
    comparison_results = {}
    
    for model_name in model_names:
        print(f"\n📊 Analyzing {model_name}...")
        analyzer = ModelAnalyzer(model_name)
        
        try:
            analyzer.load_model()
            
            if metric == "all" or metric == "architecture":
                analyzer.analyze_architecture()
            
            if metric == "all" or metric == "memory":
                analyzer.analyze_memory_usage()
            
            if metric == "all" or metric == "performance":
                analyzer.analyze_performance()
            
            comparison_results[model_name] = analyzer.analysis_results
            
        except Exception as e:
            print(f"❌ Error analyzing {model_name}: {e}")
            comparison_results[model_name] = {"error": str(e)}
    
    # Generate comparison report
    comparison_df = pd.DataFrame()
    
    for model_name, results in comparison_results.items():
        if "error" not in results and "architecture" in results:
            arch = results["architecture"]
            row_data = {
                "Model": model_name,
                "Parameters": arch["total_parameters"],
                "Model Type": arch["model_type"],
                "Hidden Size": arch["hidden_size"],
                "Attention Heads": arch["attention_heads"]
            }
            
            if "memory" in results:
                memory = results["memory"]
                if "forward_pass_memory_gb" in memory:
                    row_data["Memory (GB)"] = memory["forward_pass_memory_gb"]
            
            if "performance" in results:
                perf = results["performance"]
                if "batch_performance" in perf and 1 in perf["batch_performance"]:
                    row_data["Latency (ms)"] = perf["batch_performance"][1]["time_per_sample_ms"]
            
            comparison_df = pd.concat([comparison_df, pd.DataFrame([row_data])], ignore_index=True)
    
    print("\n📊 Model Comparison Summary:")
    print(comparison_df.to_string(index=False))
    
    return comparison_results

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="LLM Model Analyzer")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--analysis", choices=["all", "architecture", "memory", "performance"], 
                       default="all", help="Type of analysis to perform")
    parser.add_argument("--quantization", choices=["4bit", "8bit"], 
                       help="Apply quantization before analysis")
    parser.add_argument("--compare", help="Comma-separated list of models to compare")
    parser.add_argument("--metric", choices=["all", "architecture", "memory", "performance"], 
                       default="all", help="Metric for comparison")
    parser.add_argument("--output-dir", default="./analysis_results", 
                       help="Output directory for results")
    parser.add_argument("--visualize", action="store_true", 
                       help="Generate visualizations")
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple models
        model_names = [name.strip() for name in args.compare.split(",")]
        results = compare_models(model_names, args.metric)
        
        # Save comparison results
        with open(f"{args.output_dir}/model_comparison.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
    else:
        # Analyze single model
        analyzer = ModelAnalyzer(args.model)
        analyzer.load_model(args.quantization)
        
        if args.analysis == "all" or args.analysis == "architecture":
            analyzer.analyze_architecture()
        
        if args.analysis == "all" or args.analysis == "memory":
            analyzer.analyze_memory_usage()
        
        if args.analysis == "all" or args.analysis == "performance":
            analyzer.analyze_performance()
        
        # Generate outputs
        analyzer.generate_report(f"{args.output_dir}/analysis_report.json")
        
        if args.visualize:
            analyzer.generate_visualizations(args.output_dir)
        
        print(f"\n🎉 Analysis complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()