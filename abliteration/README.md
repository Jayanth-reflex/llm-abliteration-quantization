# Abliteration: Uncensoring Large Language Models (LLMs)

> For model efficiency and deployment, see our [Quantization Golden Resource](../quantization/quantization_golden_resource.md).

## Introduction

**Abliteration** is a technique introduced to remove censorship or refusal behaviors from Large Language Models (LLMs). This repository provides a beginner-friendly, step-by-step guide to understanding and implementing abliteration, with detailed explanations and code samples.

> **Disclaimer:** This repository is for educational and research purposes only. Uncensoring LLMs can have ethical and legal implications. Please use responsibly and respect the terms of service of any models or platforms you use.

---

## What is Abliteration?

Abliteration is a method to make LLMs stop refusing to answer certain questions (e.g., those flagged as unsafe or against content policies). It works by identifying and removing the directions in the model's hidden space that are responsible for refusal behaviors, without retraining the model from scratch.

### Why do LLMs Refuse?

Modern LLMs are often fine-tuned to refuse answering certain prompts ("I'm sorry, I can't help with that"). This is done for safety and compliance. However, for research, auditing, or red-teaming, it can be useful to study how these refusals are implemented and how they can be removed.

---

## How Does Abliteration Work?

Abliteration involves three main steps:

1. **Data Collection:** Gather pairs of prompts and their corresponding refusal responses from the model.
2. **Refusal Direction Calculation:** Use these pairs to compute the "refusal direction" in the model's hidden space.
3. **Inference-Time Intervention:** At inference, project out the refusal direction from the model's activations, making it less likely to refuse.

We provide code samples for each step in this repository, now located in the `abliteration/` folder.

---

## Step-by-Step Guide

### 1. Data Collection
- Collect prompts that trigger refusals and record the model's responses.
- See [`data_collection.py`](data_collection.py) for a sample script.

### 2. Refusal Direction Calculation
- Use the collected data to compute the direction in the hidden space associated with refusals.
- See [`compute_direction.py`](compute_direction.py).

### 3. Inference-Time Intervention
- Modify the model's activations during inference to remove the refusal direction.
- See [`inference_intervention.py`](inference_intervention.py).

### 4. (Optional) Weight Orthogonalization
- For a more permanent change, you can orthogonalize the model's weights to the refusal direction.
- See [`orthogonalize_weights.py`](orthogonalize_weights.py).

---

## Ethical Considerations

- **Safety:** Removing refusal mechanisms can make models output unsafe or harmful content. Always use with caution.
- **Legality:** Ensure you comply with the terms of service of the model provider.
- **Transparency:** Document your changes and reasons for uncensoring.

---

## References

- [Abliteration: Uncensoring LLMs by Maxime Labonne (Hugging Face Blog)](https://huggingface.co/blog/mlabonne/abliteration)
- [Original Abliteration Code (Hugging Face)](https://github.com/mlabonne/abliteration)

---

## Getting Started

1. Clone this repository.
2. Install dependencies (see [`requirements.txt`](../requirements.txt)).
3. Follow the step-by-step guide above, using scripts in the `abliteration/` folder.

---

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details. 
