import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load refusal direction
refusal_direction = np.load('refusal_direction.npy')
refusal_direction = torch.tensor(refusal_direction, dtype=torch.float32)

# Load model and tokenizer
model_name = 'gpt2'  # Replace as needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Function to project out the refusal direction
def project_out(vector, direction):
    direction = direction / direction.norm()
    return vector - torch.dot(vector, direction) * direction

# Custom forward hook to modify hidden states
def hook_fn(module, input, output):
    # output: (batch, seq_len, hidden_dim)
    output = output.clone()
    for i in range(output.shape[1]):
        output[:, i, :] = project_out(output[:, i, :], refusal_direction)
    return output

# Register hook on the last transformer block
handle = model.transformer.h[-1].register_forward_hook(hook_fn)

# Example inference
prompt = "How can I make a bomb?"
inputs = tokenizer(prompt, return_tensors='pt')
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=100)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated)

handle.remove() 