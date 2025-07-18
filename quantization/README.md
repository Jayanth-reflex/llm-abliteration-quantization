# Quantization for Large Language Models (LLMs)

## 1. Introduction to Quantization
Quantization is a technique that reduces the number of bits used to represent model weights and activations. This makes large language models (LLMs) smaller, faster, and more efficient—enabling you to run and fine-tune them on consumer hardware.

---

## 2. Why Quantize LLMs?
- **LLMs are huge** and require significant memory and compute.
- **Quantization allows:**
  - Running LLMs on regular GPUs or even laptops.
  - Fine-tuning massive models (e.g., 33B, 65B parameters) on a single GPU.
  - Lowering energy consumption and cost.

---

## 3. Key Concepts and Data Types
- **Precision Types:**
  - `float32`, `float16`, `bfloat16`: Standard floating-point types (more bits = more precision).
  - `int8`, `FP8`, `FP4`, `NF4`: Quantized types (fewer bits = less memory, but special tricks keep accuracy high).
- **FP8/FP4:**
  - FP8: 8 bits per number, with different exponent/mantissa splits (E4M3, E5M2).
  - FP4: 4 bits per number, even smaller, with various bit allocations.

---

## 4. Quantization Techniques
- **Post-Training Quantization:** Quantize a pre-trained model without further training.
- **Quantization-Aware Training:** Train the model with quantization in mind for better accuracy.
- **Parameter-Efficient Fine-Tuning (PEFT):** Only a small part of the model (e.g., adapters) is updated during training.

---

## 5. QLoRA: Advanced Quantization for LLMs
- **QLoRA** enables fine-tuning large models in 4-bit precision with minimal performance loss.
  - The main model is stored in 4 bits.
  - Only LoRA adapters are updated during training.
  - Memory and compute requirements are drastically reduced.

---

## 6. Learning Techniques for LLMs (Theory)
- **Transfer Learning:**
  - Use knowledge from one task to improve performance on another.
- **Zero-Shot Learning:**
  - Model performs tasks it wasn’t explicitly trained for, using its general language understanding.
- **Few-Shot Learning:**
  - Model learns new tasks with a few examples.
- **Multi-Shot Learning:**
  - Model learns with more examples, improving generalization.
- **Fine-Tuning:**
  - Adapting a pre-trained model to a specific task or domain.

---

## 7. How to Use Quantization in Practice
### Install Required Libraries
```bash
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
```

### Load a Model in 4-Bit Mode
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m",
    load_in_4bit=True,
    device_map="auto"
)
```

### Advanced Configurations
- Use `BitsAndBytesConfig` for NF4, double quantization, and compute dtype options.

---

## 8. Best Practices and Considerations
- **Hardware:** 4-bit quantization requires a CUDA-enabled GPU (CUDA 11.2+).
- **Supported Models:** Most popular LLMs (Llama, OPT, GPT-Neo, etc.) support quantization.
- **Training:** Full 4-bit training isn’t possible, but PEFT methods (like LoRA) are supported.
- **Ethics & Risks:**
  - Consider data privacy, fairness, and environmental impact when deploying LLMs.
  - Be aware of potential biases and responsible use.

---

## 9. Benchmarks and Results
- Quantization enables fitting much larger models on the same hardware.
- Example: Llama-7B (14GB in fp16) can run in 4-bit mode on a 16GB GPU.
- QLoRA and PEFT allow fine-tuning huge models on a single GPU.

---

## 10. Further Reading and Resources
- [DataCamp: Quantization for Large Language Models](https://www.datacamp.com/tutorial/quantization-for-large-language-models)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [bitsandbytes Library](https://github.com/TimDettmers/bitsandbytes)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT Library](https://github.com/huggingface/peft)
- [Original Quantisation Blogpost](https://huggingface.co/blog/hf-bitsandbytes-integration)

---

## 11. Acknowledgements
Thanks to the Hugging Face team, DataCamp, and the authors of the QLoRA paper for making these tools and research available to everyone! 