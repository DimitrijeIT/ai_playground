from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Example input
inputs = tokenizer("Hello, world!", return_tensors="pt")

# Forward pass
outputs = model(**inputs)

# # Access the last hidden state
# last_hidden_state = outputs.last_hidden_state

# # Access the pooled output
# pooler_output = outputs.pooler_output

print(outputs.pooler_output)
print(outputs.pooler_output.shape)

mask_token_logits = outputs.pooler_output[0, 0]

print(mask_token_logits)
print(mask_token_logits.shape)

# Pick the top 5 token IDs predicted for [MASK]
top_5_token_ids = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
print(top_5_token_ids)

import torch.nn.functional as F
softmax = F.softmax(mask_token_logits, dim=1)
print(softmax.shape)

# Decode the top 5 token IDs to tokens
for token_id in top_5_token_ids:
    print(tokenizer.decode([token_id]))
    print(mask_token_logits[0][token_id].item())
    print(softmax[0][token_id].item())

