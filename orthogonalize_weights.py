import torch
import numpy as np
from transformers import AutoModelForCausalLM

# Load refusal direction
refusal_direction = np.load('refusal_direction.npy')
refusal_direction = torch.tensor(refusal_direction, dtype=torch.float32)
refusal_direction = refusal_direction / refusal_direction.norm()

# Load model
model_name = 'gpt2'  # Replace as needed
model = AutoModelForCausalLM.from_pretrained(model_name)

# Orthogonalize the weights of the last transformer block as an example
def orthogonalize_weights(layer, direction):
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if param.dim() == 2:  # Only for weight matrices
                # Project each row to be orthogonal to the direction
                for i in range(param.shape[0]):
                    row = param[i]
                    proj = torch.dot(row, direction) * direction
                    param[i] = row - proj

# Apply to the last block
orthogonalize_weights(model.transformer.h[-1], refusal_direction)

# Save the modified model
model.save_pretrained('gpt2-abliterated')
print("Orthogonalized model saved to gpt2-abliterated/") 