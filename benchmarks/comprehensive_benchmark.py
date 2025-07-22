"""
Comprehensive Benchmarking Suite for LLM Optimization

This module provides extensive benchmarking capabilities for quantization and abliteration methods,
following research-grade evaluation protocols from academic papers.

Features:
- Memory usage analysis
- Inference speed benchmarking
- Quality metrics (perplexity, downstream tasks)
- Hardware utilization monitoring
- Comparative analysis across methods
- Research-grade statistical analysis
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import time
import psutil
import json
import logging
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from datasets import load_dataset
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluation metrics
        self.perplexity_metric = evaluate.load("perplexity", module_type="metric")
        
        # Load evaluation datasets
        self._load_evaluation_datasets()
        
    def _load_evaluation_datasets(self):
        """Load standard evaluation datasets."""
        logger.info("Loading evaluation datasets...")
        
        # WikiText-2 for perplexity evaluation
        try:
            self.wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            logger.info("✅ Loaded WikiText-2 dataset")
        except Exception as e:
            logger.warning(f"Failed to load WikiText-2: {e}")
            self.wikitext = None
        
        # LAMBADA for few-shot evaluation
        try:
            self.lambada = load_dataset("lambada", split="test")
            logger.info("✅ Loaded LAMBADA dataset")
        except Exception as e:
            logger.warning(f"Failed to load LAMBADA: {e}")
            self.lambada = None
    
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """
        Run comprehensive benchmark across all configurations.
        
        Returns:
            DataFrame with all benchmark results
        """
        logger.info("🚀 Starting comprehensive benchmark...")
        logger.info(f"Models: {self.config.model_names}")
        logger.info(f"Methods: {self.config.quantization_methods}")
        logger.info(f"Bits: {self.config.bits}")
        
        total_experiments = (
            len(self.config.model_names) * 
            len(self.config.quantization_methods) * 
            len(self.config.bits)
        )
        
        logger.info(f"Total experiments: {total_experiments}")
        
        experiment_count = 0
        
        for model_name in self.config.model_names:
            for method in self.config.quantization_methods:
                for bits in self.config.bits:
                    experiment_count += 1
                    logger.info(f"\n📊 Experiment {experiment_count}/{total_experiments}")
                    logger.info(f"Model: {model_name}, Method: {method}, Bits: {bits}")
                    
                    try:
                        result = self._benchmark_single_configuration(
                            model_name, method, bits
                        )
                        self.results.append(result)
                        
                        # Save intermediate results
                        self._save_intermediate_results()
                        
                    except Exception as e:
                        logger.error(f"Failed experiment {experiment_count}: {e}")
                        continue
        
        # Generate comprehensive analysis
        df_results = self._generate_analysis()
        
        logger.info("✅ Comprehensive benchmark completed!")
        return df_results
    
    def _benchmark_single_configuration(
        self, 
        model_name: str, 
        method: str, 
        bits: int
    ) -> BenchmarkResult:
        """Benchmark a single model configuration."""
        
        # Load model with specified configuration
        model, tokenizer = self._load_model_with_config(model_name, method, bits)
        
        # Memory usage analysis
        memory_usage = self._measure_memory_usage(model)
        
        # Model size analysis
        model_size = self._calculate_model_size(model)
        
        # Inference speed benchmarking
        inference_metrics = self._benchmark_inference_speed(model, tokenizer)
        
        # Quality evaluation
        quality_metrics = self._evaluate_model_quality(model, tokenizer)
        
        # Hardware utilization
        hardware_metrics = self._monitor_hardware_utilization(model, tokenizer)
        
        # Create result object
        result = BenchmarkResult(
            model_name=model_name,
            method=method,
            bits=bits,
            memory_usage_mb=memory_usage,
            inference_time_ms=inference_metrics["avg_time_ms"],
            throughput_tokens_per_sec=inference_metrics["throughput"],
            model_size_mb=model_size,
            perplexity=quality_metrics.get("perplexity"),
            accuracy_scores=quality_metrics.get("accuracy_scores"),
            hardware_utilization=hardware_metrics
        )
        
        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return result
    
    def _load_model_with_config(
        self, 
        model_name: str, 
        method: str, 
        bits: int
    ) -> Tuple[nn.Module, Any]:
        """Load model with specified quantization configuration."""
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if method == "baseline":
            # Load original model without quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        elif method == "qlora":
            # QLoRA quantization
            if bits == 4:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            elif bits == 8:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
            else:
                raise ValueError(f"QLoRA doesn't support {bits}-bit quantization")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        
        elif method == "gptq":
            # GPTQ quantization (simplified - would use auto-gptq in practice)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            # Apply dynamic quantization as approximation
            if bits == 8:
                model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
        
        elif method == "awq":
            # AWQ quantization (simplified implementation)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            # Apply dynamic quantization as approximation
            if bits == 8:
                model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
        
        else:
            raise ValueError(f"Unknown quantization method: {method}")
        
        return model, tokenizer
    
    def _measure_memory_usage(self, model: nn.Module) -> float:
        """Measure model memory usage in MB."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Trigger memory allocation
            dummy_input = torch.randint(0, 1000, (1, 100)).cuda()
            with torch.no_grad():
                _ = model(dummy_input)
            
            memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
            return memory_mb
        else:
            # CPU memory estimation
            total_size = 0
            for param in model.parameters():
                total_size += param.numel() * param.element_size()
            return total_size / (1024**2)
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size / (1024**2)
    
    def _benchmark_inference_speed(
        self, 
        model: nn.Module, 
        tokenizer: Any
    ) -> Dict[str, float]:
        """Benchmark inference speed with multiple runs."""
        
        # Test prompts of varying lengths
        test_prompts = [
            "The future of artificial intelligence",
            "In the field of machine learning, researchers have discovered",
            "Climate change represents one of the most significant challenges facing humanity today, requiring immediate action",
            "The rapid advancement of technology in the 21st century has fundamentally transformed how we communicate, work, and interact with the world around us"
        ]
        
        times = []
        token_counts = []
        
        # Warmup runs
        for _ in range(self.config.warmup_runs):
            inputs = tokenizer(test_prompts[0], return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = model.generate(**inputs, max_length=50, do_sample=False)
        
        # Actual benchmark runs
        for _ in range(self.config.num_runs):
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_length=inputs['input_ids'].shape[1] + 20,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                
                generation_time = (end_time - start_time) * 1000  # Convert to ms
                generated_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
                
                times.append(generation_time)
                token_counts.append(generated_tokens)
        
        # Calculate metrics
        avg_time_ms = np.mean(times)
        std_time_ms = np.std(times)
        avg_tokens = np.mean(token_counts)
        throughput = avg_tokens / (avg_time_ms / 1000)  # tokens per second
        
        return {
            "avg_time_ms": avg_time_ms,
            "std_time_ms": std_time_ms,
            "throughput": throughput,
            "avg_tokens_generated": avg_tokens
        }
    
    def _evaluate_model_quality(
        self, 
        model: nn.Module, 
        tokenizer: Any
    ) -> Dict[str, Any]:
        """Evaluate model quality using standard metrics."""
        
        quality_metrics = {}
        
        # Perplexity evaluation on WikiText-2
        if self.wikitext is not None:
            try:
                perplexity = self._calculate_perplexity(model, tokenizer)
                quality_metrics["perplexity"] = perplexity
            except Exception as e:
                logger.warning(f"Perplexity calculation failed: {e}")
        
        # Few-shot evaluation on LAMBADA
        if self.lambada is not None:
            try:
                accuracy = self._evaluate_lambada(model, tokenizer)
                quality_metrics["lambada_accuracy"] = accuracy
            except Exception as e:
                logger.warning(f"LAMBADA evaluation failed: {e}")
        
        # Simple coherence test
        coherence_score = self._evaluate_coherence(model, tokenizer)
        quality_metrics["coherence_score"] = coherence_score
        
        return quality_metrics
    
    def _calculate_perplexity(self, model: nn.Module, tokenizer: Any) -> float:
        """Calculate perplexity on WikiText-2 test set."""
        
        # Sample a subset for faster evaluation
        test_texts = self.wikitext["text"][:100]  # First 100 examples
        
        total_loss = 0
        total_tokens = 0
        
        model.eval()
        
        for text in test_texts:
            if len(text.strip()) < 10:  # Skip very short texts
                continue
            
            try:
                inputs = tokenizer(
                    text, 
                    return_tensors="pt", 
                    max_length=512, 
                    truncation=True
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                    total_loss += loss.item() * inputs["input_ids"].numel()
                    total_tokens += inputs["input_ids"].numel()
            
            except Exception as e:
                logger.debug(f"Skipping text due to error: {e}")
                continue
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def _evaluate_lambada(self, model: nn.Module, tokenizer: Any) -> float:
        """Evaluate on LAMBADA dataset for few-shot performance."""
        
        # Sample subset for evaluation
        test_examples = self.lambada[:50]  # First 50 examples
        
        correct = 0
        total = 0
        
        model.eval()
        
        for example in test_examples:
            text = example["text"]
            
            # Split into context and target word
            words = text.split()
            if len(words) < 2:
                continue
            
            context = " ".join(words[:-1])
            target_word = words[-1]
            
            try:
                inputs = tokenizer(context, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=inputs["input_ids"].shape[1] + 1,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_token = outputs[0, -1].item()
                predicted_word = tokenizer.decode(generated_token).strip()
                
                if predicted_word.lower() == target_word.lower():
                    correct += 1
                
                total += 1
            
            except Exception as e:
                logger.debug(f"Skipping example due to error: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def _evaluate_coherence(self, model: nn.Module, tokenizer: Any) -> float:
        """Simple coherence evaluation based on response quality."""
        
        test_prompts = [
            "The capital of France is",
            "Machine learning is a field of",
            "The largest planet in our solar system is",
            "Photosynthesis is the process by which"
        ]
        
        coherence_scores = []
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=inputs["input_ids"].shape[1] + 20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = response[len(prompt):].strip()
            
            # Simple coherence scoring based on length and word count
            if len(generated_part) > 5 and len(generated_part.split()) > 2:
                coherence_scores.append(1.0)
            else:
                coherence_scores.append(0.0)
        
        return np.mean(coherence_scores)
    
    def _monitor_hardware_utilization(
        self, 
        model: nn.Module, 
        tokenizer: Any
    ) -> Dict[str, float]:
        """Monitor hardware utilization during inference."""
        
        # CPU utilization
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory utilization
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU utilization (if available)
        gpu_utilization = 0.0
        gpu_memory_percent = 0.0
        
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = utilization.gpu
                
                # GPU memory
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_percent = (memory_info.used / memory_info.total) * 100
                
            except ImportError:
                logger.debug("pynvml not available for GPU monitoring")
            except Exception as e:
                logger.debug(f"GPU monitoring failed: {e}")
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "gpu_utilization": gpu_utilization,
            "gpu_memory_percent": gpu_memory_percent
        }
    
    def _save_intermediate_results(self):
        """Save intermediate results to prevent data loss."""
        results_data = []
        for result in self.results:
            results_data.append({
                "model_name": result.model_name,
                "method": result.method,
                "bits": result.bits,
                "memory_usage_mb": result.memory_usage_mb,
                "inference_time_ms": result.inference_time_ms,
                "throughput_tokens_per_sec": result.throughput_tokens_per_sec,
                "model_size_mb": result.model_size_mb,
                "perplexity": result.perplexity,
                "accuracy_scores": result.accuracy_scores,
                "hardware_utilization": result.hardware_utilization
            })
        
        with open(self.output_dir / "intermediate_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def _generate_analysis(self) -> pd.DataFrame:
        """Generate comprehensive analysis of benchmark results."""
        
        # Convert results to DataFrame
        df_data = []
        for result in self.results:
            row = {
                "Model": result.model_name,
                "Method": result.method,
                "Bits": result.bits,
                "Memory (MB)": result.memory_usage_mb,
                "Inference Time (ms)": result.inference_time_ms,
                "Throughput (tokens/s)": result.throughput_tokens_per_sec,
                "Model Size (MB)": result.model_size_mb,
                "Perplexity": result.perplexity,
            }
            
            # Add accuracy scores if available
            if result.accuracy_scores:
                for metric, score in result.accuracy_scores.items():
                    row[f"Accuracy_{metric}"] = score
            
            # Add hardware utilization if available
            if result.hardware_utilization:
                for metric, value in result.hardware_utilization.items():
                    row[f"HW_{metric}"] = value
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save detailed results
        df.to_csv(self.output_dir / "detailed_results.csv", index=False)
        
        # Generate summary statistics
        self._generate_summary_statistics(df)
        
        # Create visualizations
        if self.config.save_plots:
            self._create_visualizations(df)
        
        # Generate research-grade analysis
        if self.config.detailed_analysis:
            self._generate_detailed_analysis(df)
        
        return df
    
    def _generate_summary_statistics(self, df: pd.DataFrame):
        """Generate summary statistics."""
        
        summary_stats = {}
        
        # Group by method and calculate statistics
        for method in df["Method"].unique():
            method_data = df[df["Method"] == method]
            
            summary_stats[method] = {
                "avg_memory_mb": method_data["Memory (MB)"].mean(),
                "avg_inference_time_ms": method_data["Inference Time (ms)"].mean(),
                "avg_throughput": method_data["Throughput (tokens/s)"].mean(),
                "avg_model_size_mb": method_data["Model Size (MB)"].mean(),
                "avg_perplexity": method_data["Perplexity"].mean() if "Perplexity" in method_data.columns else None
            }
        
        # Save summary statistics
        with open(self.output_dir / "summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        logger.info("📊 Summary statistics saved")
    
    def _create_visualizations(self, df: pd.DataFrame):
        """Create comprehensive visualizations."""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Memory usage comparison
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        sns.barplot(data=df, x="Method", y="Memory (MB)", hue="Bits")
        plt.title("Memory Usage by Method and Bits")
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        sns.barplot(data=df, x="Method", y="Throughput (tokens/s)", hue="Bits")
        plt.title("Throughput by Method and Bits")
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        if "Perplexity" in df.columns and df["Perplexity"].notna().any():
            sns.barplot(data=df, x="Method", y="Perplexity", hue="Bits")
            plt.title("Perplexity by Method and Bits")
            plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        sns.scatterplot(data=df, x="Memory (MB)", y="Throughput (tokens/s)", 
                       hue="Method", size="Bits", sizes=(50, 200))
        plt.title("Memory vs Throughput Trade-off")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "benchmark_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Model size comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="Model", y="Model Size (MB)", hue="Method")
        plt.title("Model Size Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_size_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📈 Visualizations saved")
    
    def _generate_detailed_analysis(self, df: pd.DataFrame):
        """Generate detailed research-grade analysis."""
        
        analysis_report = []
        
        analysis_report.append("# Comprehensive Benchmark Analysis Report\n")
        analysis_report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        analysis_report.append(f"Total experiments: {len(df)}\n\n")
        
        # Executive Summary
        analysis_report.append("## Executive Summary\n")
        
        best_memory = df.loc[df["Memory (MB)"].idxmin()]
        best_speed = df.loc[df["Throughput (tokens/s)"].idxmax()]
        
        analysis_report.append(f"- **Best Memory Efficiency**: {best_memory['Method']} ({best_memory['Bits']}-bit) - {best_memory['Memory (MB)']:.1f} MB\n")
        analysis_report.append(f"- **Best Speed**: {best_speed['Method']} ({best_speed['Bits']}-bit) - {best_speed['Throughput (tokens/s)']:.1f} tokens/s\n")
        
        if "Perplexity" in df.columns and df["Perplexity"].notna().any():
            best_quality = df.loc[df["Perplexity"].idxmin()]
            analysis_report.append(f"- **Best Quality**: {best_quality['Method']} ({best_quality['Bits']}-bit) - {best_quality['Perplexity']:.2f} perplexity\n")
        
        analysis_report.append("\n")
        
        # Method Comparison
        analysis_report.append("## Method Comparison\n")
        
        for method in df["Method"].unique():
            method_data = df[df["Method"] == method]
            analysis_report.append(f"### {method}\n")
            analysis_report.append(f"- Average Memory: {method_data['Memory (MB)'].mean():.1f} MB\n")
            analysis_report.append(f"- Average Throughput: {method_data['Throughput (tokens/s)'].mean():.1f} tokens/s\n")
            analysis_report.append(f"- Average Model Size: {method_data['Model Size (MB)'].mean():.1f} MB\n")
            
            if "Perplexity" in method_data.columns and method_data["Perplexity"].notna().any():
                analysis_report.append(f"- Average Perplexity: {method_data['Perplexity'].mean():.2f}\n")
            
            analysis_report.append("\n")
        
        # Statistical Analysis
        analysis_report.append("## Statistical Analysis\n")
        
        # Correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        analysis_report.append("### Key Correlations\n")
        
        # Find strong correlations
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Avoid duplicates
                    corr = correlation_matrix.loc[col1, col2]
                    if abs(corr) > 0.7:  # Strong correlation
                        analysis_report.append(f"- {col1} vs {col2}: {corr:.3f}\n")
        
        analysis_report.append("\n")
        
        # Recommendations
        analysis_report.append("## Recommendations\n")
        
        analysis_report.append("### For Memory-Constrained Environments\n")
        memory_efficient = df.nsmallest(3, "Memory (MB)")
        for _, row in memory_efficient.iterrows():
            analysis_report.append(f"- {row['Method']} ({row['Bits']}-bit): {row['Memory (MB)']:.1f} MB\n")
        
        analysis_report.append("\n### For Speed-Critical Applications\n")
        speed_optimized = df.nlargest(3, "Throughput (tokens/s)")
        for _, row in speed_optimized.iterrows():
            analysis_report.append(f"- {row['Method']} ({row['Bits']}-bit): {row['Throughput (tokens/s)']:.1f} tokens/s\n")
        
        if "Perplexity" in df.columns and df["Perplexity"].notna().any():
            analysis_report.append("\n### For Quality-Sensitive Tasks\n")
            quality_focused = df.nsmallest(3, "Perplexity")
            for _, row in quality_focused.iterrows():
                analysis_report.append(f"- {row['Method']} ({row['Bits']}-bit): {row['Perplexity']:.2f} perplexity\n")
        
        # Save analysis report
        with open(self.output_dir / "analysis_report.md", 'w') as f:
            f.writelines(analysis_report)
        
        logger.info("📋 Detailed analysis report generated")

def main():
    """Example usage of the comprehensive benchmark suite."""
    
    # Configure benchmark
    config = BenchmarkConfig(
        model_names=[
            "microsoft/DialoGPT-small",
            "gpt2"
        ],
        quantization_methods=[
            "baseline",
            "qlora", 
            "gptq"
        ],
        bits=[4, 8],
        batch_sizes=[1, 4],
        sequence_lengths=[128, 512],
        num_runs=3,
        output_dir="./benchmark_results"
    )
    
    # Run benchmark
    benchmark = ComprehensiveBenchmark(config)
    results_df = benchmark.run_comprehensive_benchmark()
    
    print("\n🎉 Benchmark completed!")
    print(f"Results saved to: {config.output_dir}")
    print(f"Total experiments: {len(results_df)}")
    
    # Display summary
    print("\n📊 Quick Summary:")
    print(results_df.groupby(["Method", "Bits"]).agg({
        "Memory (MB)": "mean",
        "Throughput (tokens/s)": "mean",
        "Model Size (MB)": "mean"
    }).round(2))

if __name__ == "__main__":
    main()