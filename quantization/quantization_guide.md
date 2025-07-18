# Comprehensive Quantization Guide for Large Language Models (LLMs)

> **A detailed reference merging DataCamp, Hugging Face, and QLoRA resources.**

---

## Table of Contents
1. Introduction
2. Why Quantization Matters for LLMs
3. The Scale and Cost of LLMs
4. What is Quantization?
5. Data Types and Precision (FP32, FP16, INT8, FP8, FP4, NF4)
6. Types of Quantization
    - Static Quantization
    - Dynamic Quantization
    - Post-Training Quantization (PTQ)
    - Quantization-Aware Training (QAT)
    - Binary/Ternary Quantization
7. Advanced Quantization: QLoRA and Modern Methods
8. Practical Usage: Hugging Face Transformers & bitsandbytes
9. Best Practices, Benefits, and Challenges
10. Benchmarks and Real-World Results
11. Visuals & Diagrams
12. Further Reading and References
13. Acknowledgements

---

## 1. Introduction
Large Language Models (LLMs) are at the forefront of AI, but their size and complexity make them resource-intensive. Quantization is a key technique to make LLMs more accessible, efficient, and deployable on consumer hardware.

---

## 2. Why Quantization Matters for LLMs
- LLMs like GPT-4 have up to 175 billion parameters, requiring massive memory and compute.
- Quantization reduces the memory, storage, and compute requirements, enabling:
  - Running LLMs on regular GPUs, laptops, and edge devices.
  - Fine-tuning massive models (e.g., 33B, 65B parameters) on a single GPU.
  - Lowering energy consumption and cost.

---

## 3. The Scale and Cost of LLMs
- **Bar Graph Example:**
  ![Bar graph of LLM parameter sizes](https://cdn.datacamp.com/tutorial_images/llm-quantization-bar-graph.png)
- Running LLMs is expensive due to hardware (GPUs, accelerators) and energy costs.
- Quantization enables on-premises and edge deployments, reducing operational costs.

---

## 4. What is Quantization?
- Quantization reduces the numerical precision of model parameters (e.g., from 32-bit floats to 8-bit or 4-bit integers).
- This is like compressing a high-res image to a lower resolution: smaller size, but key features are preserved.
- Allows LLMs to run on less powerful hardware with minimal performance loss.
- **Diagram Example:**
  ![Floating point to integer quantization](https://cdn.datacamp.com/tutorial_images/float-to-int-quantization.png)

---

## 5. Data Types and Precision
- **float32, float16, bfloat16:** Standard floating-point types (more bits = more precision).
- **int8, FP8, FP4, NF4:** Quantized types (fewer bits = less memory, but special tricks keep accuracy high).
- **FP8:** 8 bits per number, with E4M3 (4 exponent, 3 mantissa) and E5M2 (5 exponent, 2 mantissa) formats.
  ![FP8 format](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bitsandbytes/FP8-scheme.png)
- **FP4:** 4 bits per number, with various bit allocations. No fixed format; combinations of exponent and mantissa bits are possible.

---

## 6. Types of Quantization
### Static Quantization
- Applied during training; weights and activations are quantized and fixed for all layers.
- Pros: Predictable memory use, good for edge devices.
- Cons: Less adaptable to varying input patterns.

### Dynamic Quantization
- Weights are quantized ahead of time; activations are quantized dynamically during inference.
- Pros: Balances compression and runtime efficiency.
- Cons: Slightly more computational overhead.

### Post-Training Quantization (PTQ)
- Quantization is applied after training, often without retraining.
- Pros: Fast, reduces model size, improves inference speed.
- Cons: May reduce accuracy, needs calibration.

### Quantization-Aware Training (QAT)
- Model is trained with quantization in mind, learning to handle quantization errors.
- Pros: Preserves accuracy, robust to low precision.
- Cons: Requires retraining, more compute during training.

### Binary/Ternary Quantization
- Weights are quantized to two (binary: -1, 1) or three (ternary: -1, 0, 1) values.
- Pros: Maximum compression and speed.
- Cons: Significant accuracy loss, not suitable for all tasks.

---

## 7. Advanced Quantization: QLoRA and Modern Methods
- **QLoRA**: Efficient fine-tuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance.
  - Uses 4-bit NormalFloat (NF4), double quantization, and paged optimizers.
  - Only LoRA adapters are updated during training; main model is frozen in 4 bits.
  - [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- **GPTQ**: Layer-wise, asymmetric quantization for 4-bit models, using inverse-Hessian weighting for error minimization.
- **GGUF**: Allows offloading layers to CPU, using block-wise quantization for flexible deployment.
- **BitNet**: 1-bit and 1.58-bit quantization for extreme compression, using signum and absmean quantization.

---

## 8. Practical Usage: Hugging Face Transformers & bitsandbytes
### Getting Started
```bash
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
```

### Loading a Model in 4-Bit Mode
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m",
    load_in_4bit=True,
    device_map="auto"
)
```

### Advanced Usage
- Use `BitsAndBytesConfig` for NF4, double quantization, and compute dtype options.
- Example:
```python
from transformers import BitsAndBytesConfig
import torch
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
```

---

## 9. Best Practices, Benefits, and Challenges
### Benefits
- Reduces memory and storage requirements.
- Enables LLMs on edge devices and consumer hardware.
- Improves inference speed and energy efficiency.
- Democratizes access to advanced AI.

### Challenges
- Can introduce accuracy loss (quantization error).
- Requires careful calibration and sometimes retraining.
- Not all models or tasks are equally robust to quantization.
- Hardware requirements: 4-bit quantization requires a CUDA-enabled GPU (CUDA 11.2+).

---

## 10. Benchmarks and Real-World Results
- Quantization enables fitting much larger models on the same hardware.
- Example: Llama-7B (14GB in fp16) can run in 4-bit mode on a 16GB GPU.
- QLoRA and PEFT allow fine-tuning huge models on a single GPU.
- **Benchmark Table Example:**

| Model name                          | Half precision model size (in GB) | Hardware type / total VRAM | quantization method (CD=compute dtype / GC=gradient checkpointing / NQ=nested quantization) | batch_size | gradient accumulation steps | optimizer         | seq_len | Result |
| ----------------------------------- | --------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------- | ---------- | --------------------------- | ----------------- | ------- | ------ |
| decapoda-research/llama-7b-hf       | 14GB                              | 1xNVIDIA-T4 / 16GB         | 4bit + NF4 + bf16 CD + no GC                                                                | 1          | 4                           | AdamW             | 512     | **No OOM** |
| decapoda-research/llama-13b-hf      | 27GB                              | 2xNVIDIA-T4 / 32GB         | 4bit + NF4 + fp16 CD + GC + NQ                                                              | 1          | 4                           | AdamW             | 1024    | **No OOM** |

---

## 11. Visuals & Diagrams
- **Parameter Size Bar Graph:**
  ![Bar graph of LLM parameter sizes](https://cdn.datacamp.com/tutorial_images/llm-quantization-bar-graph.png)
- **FP8 Format:**
  ![FP8 format](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bitsandbytes/FP8-scheme.png)
- **Float to Int Quantization:**
  ![Floating point to integer quantization](https://cdn.datacamp.com/tutorial_images/float-to-int-quantization.png)
- **Static vs Dynamic Quantization:**
  ![Static vs Dynamic Quantization](https://cdn.datacamp.com/tutorial_images/static-vs-dynamic-quantization.png)
- **Quantization Error Example:**
  ![Quantization error illustration](https://cdn.datacamp.com/tutorial_images/quantization-error.png)
- **LoRA Adapter Animation:**
  ![LoRA Adapter Animation](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/133_trl_peft/lora-animated.gif)

---

## 12. Further Reading and References
- [DataCamp Quantization for Large Language Models Tutorial](https://www.datacamp.com/tutorial/quantization-for-large-language-models)
- [Hugging Face Quantization Blog](https://huggingface.co/blog/hf-bitsandbytes-integration)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
- [Original repository for replicating QLoRA results](https://github.com/artidoro/qlora)
- [Guanaco 33b playground](https://huggingface.co/spaces/uwnlp/guanaco-playground-tgi)

---

## 13. Acknowledgements
Thanks to the Hugging Face team, DataCamp, the QLoRA authors, and the open-source community for making these tools and research available to everyone! 