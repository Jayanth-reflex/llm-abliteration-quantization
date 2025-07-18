import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from tqdm import tqdm

# Load refusal data
with open('refusal_data.json', 'r') as f:
    data = json.load(f)

# Load model and tokenizer (replace with your model)
model_name = 'gpt2'  # Replace as needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

hidden_states = []

for item in tqdm(data, desc="Extracting hidden states"):
    prompt = item['prompt']
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Use the last hidden state of the last token
        last_hidden = outputs.hidden_states[-1][0, -1, :].numpy()
        hidden_states.append(last_hidden)

# Compute principal component (refusal direction)
pca = PCA(n_components=1)
pca.fit(hidden_states)
refusal_direction = pca.components_[0]

# Save the refusal direction
np.save('refusal_direction.npy', refusal_direction)
print("Refusal direction saved to refusal_direction.npy.") 