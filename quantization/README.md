# Quantization for Large Language Models (LLMs)

## What is Quantization?

Quantization is a technique that makes large language models (LLMs) smaller and faster by reducing the number of bits used to store their weights. This helps you run or train big models on regular computers or GPUs with less memory.

---

## Why Quantize LLMs?

- **LLMs are huge!** They need a lot of memory and computing power.
- **Quantization lets you use LLMs on consumer hardware** (like a single GPU or even a laptop).
- **You can even fine-tune big models** (like 33B or 65B parameters) on a single GPU using quantization.

---

## Key Concepts (Explained Simply)

### 1. Data Types
- **float32, float16, bfloat16:** These are common ways to store numbers in computers. More bits = more precision, but also more memory.
- **int8, 4-bit (FP4, NF4):** Fewer bits = less memory, but also less precision. Special tricks help keep accuracy high.

### 2. FP8 and FP4
- **FP8:** 8 bits per number. Used for some deep learning tasks. Two main types: E4M3 (4 exponent, 3 mantissa bits) and E5M2 (5 exponent, 2 mantissa bits).
- **FP4:** 4 bits per number. Even smaller! There are different ways to split the bits between sign, exponent, and mantissa.

### 3. QLoRA
- **QLoRA** is a method that lets you fine-tune big models using 4-bit quantization, while keeping almost the same performance as full-precision training.
- **How?**
  - The main model is stored in 4 bits (very small!).
  - Only a small part (called LoRA adapters) is updated during training.
  - This saves a lot of memory and lets you train huge models on a single GPU.

---

## How to Use Quantization in Transformers (Step-by-Step)

### 1. Install the Required Libraries
- You need the latest versions of these libraries:
  - `bitsandbytes`
  - `transformers`
  - `peft`
  - `accelerate`
- Install them with:

```bash
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
```

### 2. Load a Model in 4-Bit Mode
- Use the `load_in_4bit=True` argument:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m",
    load_in_4bit=True,
    device_map="auto"
)
```
- That's it! The model is now loaded in 4-bit mode.

### 3. Advanced Options
- You can use different quantization types (like NF4 or FP4), double quantization, and change the compute type (float16, bfloat16, etc.)
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

model_nf4 = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=nf4_config
)
```

---

## Common Questions (FAQ)

- **Q: Can I use 4-bit quantization on any hardware?**
  - A: You need a GPU with CUDA 11.2 or higher. CPUs are not supported for 4-bit quantization.

- **Q: Which models are supported?**
  - A: Most popular models like Llama, OPT, GPT-Neo, GPT-NeoX, and more. If your model supports `device_map` in `from_pretrained`, it likely works.

- **Q: Can I train a model in 4-bit?**
  - A: You can't train the whole model in 4-bit, but you can train adapters (like LoRA) on top of a 4-bit model. This is called parameter-efficient fine-tuning (PEFT).

---

## Benchmarks and Results

- Quantization lets you fit much bigger models on the same hardware.
- Example: Llama-7B (14GB in fp16) can run in 4-bit mode on a 16GB GPU with no out-of-memory errors.
- You can fine-tune huge models on a single GPU using QLoRA and PEFT.

---

## Resources and References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [bitsandbytes Library](https://github.com/TimDettmers/bitsandbytes)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT Library](https://github.com/huggingface/peft)
- [Original Quantisation Blogpost](https://huggingface.co/blog/hf-bitsandbytes-integration)

---

## Acknowledgements

Thanks to the Hugging Face team and the authors of the QLoRA paper for making these tools and research available to everyone! 