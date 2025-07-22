"""
Interactive Model Comparison Dashboard

A Streamlit-based interactive dashboard for comparing different quantization methods
and their effects on model performance. Useful for beginners to understand trade-offs
and for advanced users to make informed decisions.

Features:
- Real-time model comparison
- Interactive visualizations
- Performance metrics
- Quality assessment
- Hardware utilization monitoring
- Export capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import psutil
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="LLM Optimization Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .comparison-table {
        font-size: 0.9rem;
    }
    .sidebar-section {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class ModelComparisonDashboard:
    """Interactive dashboard for model comparison."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.models_cache = {}
        self.results_cache = {}
        
        # Available models (small ones for demo)
        self.available_models = {
            "microsoft/DialoGPT-small": "DialoGPT Small (117M)",
            "gpt2": "GPT-2 (124M)",
            "microsoft/DialoGPT-medium": "DialoGPT Medium (345M)",
            "gpt2-medium": "GPT-2 Medium (355M)"
        }
        
        # Available quantization methods
        self.quantization_methods = {
            "baseline": "No Quantization (FP16)",
            "8bit": "8-bit Quantization",
            "4bit": "4-bit Quantization (QLoRA)",
            "4bit_nf4": "4-bit NF4 (Advanced)"
        }
    
    def run_dashboard(self):
        """Run the main dashboard."""
        
        # Header
        st.markdown('<h1 class="main-header">🚀 LLM Optimization Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        **Compare different quantization methods and their effects on model performance.**
        Select models and methods from the sidebar to start comparing!
        """)
        
        # Sidebar configuration
        self._render_sidebar()
        
        # Main content area
        if st.session_state.get('models_to_compare'):
            self._render_main_content()
        else:
            self._render_welcome_screen()
    
    def _render_sidebar(self):
        """Render the sidebar with configuration options."""
        
        st.sidebar.markdown("## 🔧 Configuration")
        
        # Model selection
        st.sidebar.markdown("### Select Models")
        selected_models = st.sidebar.multiselect(
            "Choose models to compare:",
            options=list(self.available_models.keys()),
            default=["microsoft/DialoGPT-small"],
            format_func=lambda x: self.available_models[x]
        )
        
        # Quantization method selection
        st.sidebar.markdown("### Select Methods")
        selected_methods = st.sidebar.multiselect(
            "Choose quantization methods:",
            options=list(self.quantization_methods.keys()),
            default=["baseline", "8bit"],
            format_func=lambda x: self.quantization_methods[x]
        )
        
        # Advanced options
        st.sidebar.markdown("### Advanced Options")
        
        test_prompts = st.sidebar.text_area(
            "Test Prompts (one per line):",
            value="Hello, how are you?\\nExplain artificial intelligence\\nTell me a joke",
            height=100
        ).split('\\n')
        
        max_length = st.sidebar.slider(
            "Max Generation Length:",
            min_value=20,
            max_value=200,
            value=50,
            step=10
        )
        
        num_runs = st.sidebar.slider(
            "Number of Test Runs:",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )
        
        # Store in session state
        st.session_state.models_to_compare = selected_models
        st.session_state.methods_to_compare = selected_methods
        st.session_state.test_prompts = test_prompts
        st.session_state.max_length = max_length
        st.session_state.num_runs = num_runs
        
        # Action buttons
        st.sidebar.markdown("### Actions")
        
        if st.sidebar.button("🚀 Run Comparison", type="primary"):
            st.session_state.run_comparison = True
        
        if st.sidebar.button("🗑️ Clear Cache"):
            self.models_cache.clear()
            self.results_cache.clear()
            st.success("Cache cleared!")
        
        if st.sidebar.button("💾 Export Results"):
            self._export_results()
    
    def _render_welcome_screen(self):
        """Render welcome screen when no models are selected."""
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ## 👋 Welcome to the LLM Optimization Dashboard!
            
            This interactive tool helps you:
            
            ### 🔍 **Compare Models**
            - Select multiple language models
            - Test different quantization methods
            - See real-time performance metrics
            
            ### 📊 **Analyze Performance**
            - Memory usage comparison
            - Inference speed benchmarks
            - Quality assessment
            - Hardware utilization
            
            ### 🎯 **Make Informed Decisions**
            - Understand trade-offs
            - Choose optimal configurations
            - Export results for reports
            
            ---
            
            **Get Started:**
            1. Select models from the sidebar
            2. Choose quantization methods
            3. Click "Run Comparison"
            4. Explore the results!
            """)
            
            # Quick start examples
            st.markdown("### 🚀 Quick Start Examples")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("📱 Mobile Deployment", help="Optimize for mobile/edge deployment"):
                    st.session_state.models_to_compare = ["microsoft/DialoGPT-small"]
                    st.session_state.methods_to_compare = ["4bit", "8bit"]
                    st.rerun()
            
            with col_b:
                if st.button("⚡ Speed Comparison", help="Compare inference speeds"):
                    st.session_state.models_to_compare = ["gpt2", "microsoft/DialoGPT-small"]
                    st.session_state.methods_to_compare = ["baseline", "8bit", "4bit"]
                    st.rerun()
    
    def _render_main_content(self):
        """Render the main content area with comparisons."""
        
        # Run comparison if requested
        if st.session_state.get('run_comparison', False):
            self._run_comparison()
            st.session_state.run_comparison = False
        
        # Display results if available
        if hasattr(st.session_state, 'comparison_results'):
            self._display_comparison_results()
        else:
            st.info("👆 Click 'Run Comparison' in the sidebar to start!")
    
    def _run_comparison(self):
        """Run the model comparison."""
        
        models = st.session_state.models_to_compare
        methods = st.session_state.methods_to_compare
        
        if not models or not methods:
            st.error("Please select at least one model and one method!")
            return
        
        # Progress tracking
        total_combinations = len(models) * len(methods)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        current_step = 0
        
        for model_name in models:
            for method in methods:
                current_step += 1
                progress = current_step / total_combinations
                
                status_text.text(f"Testing {self.available_models[model_name]} with {self.quantization_methods[method]}...")
                progress_bar.progress(progress)
                
                try:
                    result = self._benchmark_model_method(model_name, method)
                    results.append(result)
                except Exception as e:
                    st.error(f"Error testing {model_name} with {method}: {str(e)}")
                    continue
        
        # Store results
        st.session_state.comparison_results = results
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Comparison completed! Tested {len(results)} configurations.")
    
    def _benchmark_model_method(self, model_name: str, method: str) -> Dict:
        """Benchmark a specific model with a specific method."""
        
        # Check cache first
        cache_key = f"{model_name}_{method}"
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
        
        # Load model
        model, tokenizer = self._load_model(model_name, method)
        
        # Run benchmarks
        result = {
            "model_name": model_name,
            "model_display_name": self.available_models[model_name],
            "method": method,
            "method_display_name": self.quantization_methods[method],
            "model_size_mb": self._calculate_model_size(model),
            "memory_usage_mb": self._measure_memory_usage(model),
            "inference_metrics": self._benchmark_inference(model, tokenizer),
            "quality_metrics": self._assess_quality(model, tokenizer),
            "hardware_metrics": self._monitor_hardware(model, tokenizer)
        }
        
        # Cache result
        self.results_cache[cache_key] = result
        
        # Cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return result
    
    def _load_model(self, model_name: str, method: str) -> Tuple:
        """Load model with specified quantization method."""
        
        cache_key = f"{model_name}_{method}"
        if cache_key in self.models_cache:
            return self.models_cache[cache_key]
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if method == "baseline":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        elif method == "8bit":
            config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=config,
                device_map="auto"
            )
        
        elif method == "4bit":
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=config,
                device_map="auto"
            )
        
        elif method == "4bit_nf4":
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=config,
                device_map="auto"
            )
        
        # Cache the loaded model
        self.models_cache[cache_key] = (model, tokenizer)
        
        return model, tokenizer
    
    def _calculate_model_size(self, model) -> float:
        """Calculate model size in MB."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size / (1024 ** 2)
    
    def _measure_memory_usage(self, model) -> float:
        """Measure memory usage in MB."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Trigger memory allocation
            dummy_input = torch.randint(0, 1000, (1, 50))
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            return self._calculate_model_size(model)
    
    def _benchmark_inference(self, model, tokenizer) -> Dict:
        """Benchmark inference performance."""
        
        test_prompts = st.session_state.get('test_prompts', ["Hello, how are you?"])
        max_length = st.session_state.get('max_length', 50)
        num_runs = st.session_state.get('num_runs', 3)
        
        times = []
        token_counts = []
        
        for _ in range(num_runs):
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=max_length,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                end_time = time.time()
                
                generation_time = (end_time - start_time) * 1000  # ms
                generated_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
                
                times.append(generation_time)
                token_counts.append(generated_tokens)
        
        return {
            "avg_time_ms": np.mean(times),
            "std_time_ms": np.std(times),
            "avg_tokens": np.mean(token_counts),
            "throughput_tokens_per_sec": np.mean(token_counts) / (np.mean(times) / 1000)
        }
    
    def _assess_quality(self, model, tokenizer) -> Dict:
        """Assess model quality with simple metrics."""
        
        test_prompts = [
            "The capital of France is",
            "Machine learning is",
            "The largest planet is"
        ]
        
        responses = []
        coherence_scores = []
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = response[len(prompt):].strip()
            
            responses.append(generated_part)
            
            # Simple coherence scoring
            if len(generated_part) > 3 and len(generated_part.split()) > 1:
                coherence_scores.append(1.0)
            else:
                coherence_scores.append(0.0)
        
        return {
            "sample_responses": responses,
            "avg_coherence": np.mean(coherence_scores),
            "avg_response_length": np.mean([len(r.split()) for r in responses])
        }
    
    def _monitor_hardware(self, model, tokenizer) -> Dict:
        """Monitor hardware utilization."""
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # GPU utilization (simplified)
        gpu_utilization = 0.0
        if torch.cuda.is_available():
            gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0.0
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "gpu_utilization": gpu_utilization
        }
    
    def _display_comparison_results(self):
        """Display the comparison results."""
        
        results = st.session_state.comparison_results
        
        if not results:
            st.warning("No results to display!")
            return
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "⚡ Performance", "🎯 Quality", "💾 Details"])
        
        with tab1:
            self._display_overview(results)
        
        with tab2:
            self._display_performance_analysis(results)
        
        with tab3:
            self._display_quality_analysis(results)
        
        with tab4:
            self._display_detailed_results(results)
    
    def _display_overview(self, results: List[Dict]):
        """Display overview of results."""
        
        st.markdown("## 📊 Comparison Overview")
        
        # Create summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_memory = min(r['memory_usage_mb'] for r in results)
            st.metric("Best Memory Usage", f"{min_memory:.1f} MB")
        
        with col2:
            max_throughput = max(r['inference_metrics']['throughput_tokens_per_sec'] for r in results)
            st.metric("Best Throughput", f"{max_throughput:.1f} tok/s")
        
        with col3:
            min_size = min(r['model_size_mb'] for r in results)
            st.metric("Smallest Model", f"{min_size:.1f} MB")
        
        with col4:
            avg_coherence = np.mean([r['quality_metrics']['avg_coherence'] for r in results])
            st.metric("Avg Coherence", f"{avg_coherence:.2f}")
        
        # Summary table
        st.markdown("### 📋 Summary Table")
        
        summary_data = []
        for result in results:
            summary_data.append({
                "Model": result['model_display_name'],
                "Method": result['method_display_name'],
                "Size (MB)": f"{result['model_size_mb']:.1f}",
                "Memory (MB)": f"{result['memory_usage_mb']:.1f}",
                "Speed (tok/s)": f"{result['inference_metrics']['throughput_tokens_per_sec']:.1f}",
                "Coherence": f"{result['quality_metrics']['avg_coherence']:.2f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Quick insights
        st.markdown("### 💡 Quick Insights")
        
        # Find best configurations
        best_memory = min(results, key=lambda x: x['memory_usage_mb'])
        best_speed = max(results, key=lambda x: x['inference_metrics']['throughput_tokens_per_sec'])
        best_quality = max(results, key=lambda x: x['quality_metrics']['avg_coherence'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"🏆 **Memory Champion**\\n{best_memory['model_display_name']} with {best_memory['method_display_name']}")
        
        with col2:
            st.success(f"⚡ **Speed Champion**\\n{best_speed['model_display_name']} with {best_speed['method_display_name']}")
        
        with col3:
            st.success(f"🎯 **Quality Champion**\\n{best_quality['model_display_name']} with {best_quality['method_display_name']}")
    
    def _display_performance_analysis(self, results: List[Dict]):
        """Display performance analysis."""
        
        st.markdown("## ⚡ Performance Analysis")
        
        # Prepare data for visualization
        perf_data = []
        for result in results:
            perf_data.append({
                "Configuration": f"{result['model_display_name']}\\n{result['method_display_name']}",
                "Model": result['model_display_name'],
                "Method": result['method_display_name'],
                "Memory Usage (MB)": result['memory_usage_mb'],
                "Model Size (MB)": result['model_size_mb'],
                "Throughput (tokens/s)": result['inference_metrics']['throughput_tokens_per_sec'],
                "Avg Time (ms)": result['inference_metrics']['avg_time_ms']
            })
        
        df_perf = pd.DataFrame(perf_data)
        
        # Memory vs Throughput scatter plot
        fig = px.scatter(
            df_perf,
            x="Memory Usage (MB)",
            y="Throughput (tokens/s)",
            color="Method",
            size="Model Size (MB)",
            hover_name="Configuration",
            title="Memory Usage vs Throughput Trade-off"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance comparison bars
        col1, col2 = st.columns(2)
        
        with col1:
            fig_memory = px.bar(
                df_perf,
                x="Configuration",
                y="Memory Usage (MB)",
                color="Method",
                title="Memory Usage Comparison"
            )
            fig_memory.update_xaxis(tickangle=45)
            st.plotly_chart(fig_memory, use_container_width=True)
        
        with col2:
            fig_speed = px.bar(
                df_perf,
                x="Configuration",
                y="Throughput (tokens/s)",
                color="Method",
                title="Throughput Comparison"
            )
            fig_speed.update_xaxis(tickangle=45)
            st.plotly_chart(fig_speed, use_container_width=True)
    
    def _display_quality_analysis(self, results: List[Dict]):
        """Display quality analysis."""
        
        st.markdown("## 🎯 Quality Analysis")
        
        # Sample responses
        st.markdown("### 📝 Sample Responses")
        
        for i, result in enumerate(results):
            with st.expander(f"{result['model_display_name']} - {result['method_display_name']}"):
                responses = result['quality_metrics']['sample_responses']
                for j, response in enumerate(responses):
                    st.write(f"**Response {j+1}:** {response}")
                
                st.write(f"**Coherence Score:** {result['quality_metrics']['avg_coherence']:.2f}")
                st.write(f"**Avg Response Length:** {result['quality_metrics']['avg_response_length']:.1f} words")
        
        # Quality metrics comparison
        quality_data = []
        for result in results:
            quality_data.append({
                "Configuration": f"{result['model_display_name']}\\n{result['method_display_name']}",
                "Coherence Score": result['quality_metrics']['avg_coherence'],
                "Avg Response Length": result['quality_metrics']['avg_response_length']
            })
        
        df_quality = pd.DataFrame(quality_data)
        
        fig_quality = px.bar(
            df_quality,
            x="Configuration",
            y="Coherence Score",
            title="Quality Comparison (Coherence Score)"
        )
        fig_quality.update_xaxis(tickangle=45)
        st.plotly_chart(fig_quality, use_container_width=True)
    
    def _display_detailed_results(self, results: List[Dict]):
        """Display detailed results."""
        
        st.markdown("## 💾 Detailed Results")
        
        # Detailed metrics table
        detailed_data = []
        for result in results:
            detailed_data.append({
                "Model": result['model_display_name'],
                "Method": result['method_display_name'],
                "Model Size (MB)": result['model_size_mb'],
                "Memory Usage (MB)": result['memory_usage_mb'],
                "Avg Time (ms)": result['inference_metrics']['avg_time_ms'],
                "Std Time (ms)": result['inference_metrics']['std_time_ms'],
                "Throughput (tok/s)": result['inference_metrics']['throughput_tokens_per_sec'],
                "Coherence": result['quality_metrics']['avg_coherence'],
                "Response Length": result['quality_metrics']['avg_response_length'],
                "CPU %": result['hardware_metrics']['cpu_percent'],
                "Memory %": result['hardware_metrics']['memory_percent']
            })
        
        df_detailed = pd.DataFrame(detailed_data)
        st.dataframe(df_detailed, use_container_width=True)
        
        # Export options
        st.markdown("### 📤 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df_detailed.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV",
                data=csv,
                file_name="model_comparison_results.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name="model_comparison_results.json",
                mime="application/json"
            )
        
        with col3:
            if st.button("📊 Generate Report"):
                self._generate_report(results)
    
    def _export_results(self):
        """Export results to files."""
        if hasattr(st.session_state, 'comparison_results'):
            results = st.session_state.comparison_results
            
            # Create export directory
            export_dir = Path("./dashboard_exports")
            export_dir.mkdir(exist_ok=True)
            
            # Export CSV
            detailed_data = []
            for result in results:
                detailed_data.append({
                    "Model": result['model_display_name'],
                    "Method": result['method_display_name'],
                    "Model_Size_MB": result['model_size_mb'],
                    "Memory_Usage_MB": result['memory_usage_mb'],
                    "Avg_Time_ms": result['inference_metrics']['avg_time_ms'],
                    "Throughput_tokens_per_sec": result['inference_metrics']['throughput_tokens_per_sec'],
                    "Coherence_Score": result['quality_metrics']['avg_coherence']
                })
            
            df = pd.DataFrame(detailed_data)
            df.to_csv(export_dir / "comparison_results.csv", index=False)
            
            # Export JSON
            with open(export_dir / "comparison_results.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            st.success(f"Results exported to {export_dir}")
        else:
            st.warning("No results to export. Run a comparison first!")
    
    def _generate_report(self, results: List[Dict]):
        """Generate a comprehensive report."""
        
        report_lines = []
        report_lines.append("# LLM Optimization Comparison Report\\n")
        report_lines.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        # Executive Summary
        report_lines.append("## Executive Summary\\n")
        report_lines.append(f"- Total configurations tested: {len(results)}\\n")
        
        best_memory = min(results, key=lambda x: x['memory_usage_mb'])
        best_speed = max(results, key=lambda x: x['inference_metrics']['throughput_tokens_per_sec'])
        
        report_lines.append(f"- Best memory efficiency: {best_memory['model_display_name']} with {best_memory['method_display_name']} ({best_memory['memory_usage_mb']:.1f} MB)\\n")
        report_lines.append(f"- Best speed: {best_speed['model_display_name']} with {best_speed['method_display_name']} ({best_speed['inference_metrics']['throughput_tokens_per_sec']:.1f} tokens/s)\\n\\n")
        
        # Detailed Results
        report_lines.append("## Detailed Results\\n")
        for result in results:
            report_lines.append(f"### {result['model_display_name']} - {result['method_display_name']}\\n")
            report_lines.append(f"- Model Size: {result['model_size_mb']:.1f} MB\\n")
            report_lines.append(f"- Memory Usage: {result['memory_usage_mb']:.1f} MB\\n")
            report_lines.append(f"- Throughput: {result['inference_metrics']['throughput_tokens_per_sec']:.1f} tokens/s\\n")
            report_lines.append(f"- Coherence Score: {result['quality_metrics']['avg_coherence']:.2f}\\n\\n")
        
        report_content = "".join(report_lines)
        
        st.download_button(
            label="📄 Download Report",
            data=report_content,
            file_name="llm_optimization_report.md",
            mime="text/markdown"
        )

def main():
    """Main function to run the dashboard."""
    dashboard = ModelComparisonDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()