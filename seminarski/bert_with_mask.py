# from transformers import AutoModel, AutoTokenizer
# import torch

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# model = AutoModel.from_pretrained('bert-base-uncased')

# # Prepare input
# text = "Paris is the [MASK] of France."
# inputs = tokenizer(text, return_tensors="pt")

# # Find the position of the [MASK] token
# mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

# # Forward pass
# with torch.no_grad():
#     outputs = model(**inputs)

# # Get the logits for the masked token
# mask_token_logits = outputs.last_hidden_state[0, mask_token_index, :]

# # Pick the top 5 token IDs predicted for [MASK]
# top_5_token_ids = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

# # Decode the top 5 token IDs to tokens
# for token_id in top_5_token_ids:
#     print(tokenizer.decode([token_id]))



from transformers import BertTokenizer, BertForMaskedLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load tokenizer and model
tokenizer = .from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.to(device)

print("Masked ID = %d", tokenizer.mask_token_id)
# Prepare input
text = "Capital of Serbia is [MASK] [MASK] [MASK] [MASK]."
inputs = tokenizer(text, return_tensors="pt")
print(inputs)
inputs = {k:v.to(device) for k,v in inputs.items()}
# Identify the position of the [MASK] token
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
sep_token_index = torch.where(inputs["input_ids"] == 102)[1]
print("Masked token index", mask_token_index)
print("Sep token index", sep_token_index)
decoded = {tokenizer.decode(i) for i in inputs['input_ids']}
print(decoded)
print("token id for blegrade 10291, ", tokenizer.decode([10291]))
# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Get logits for the masked token
mask_token_logits = outputs.logits[0, mask_token_index, :]
# mask_token_logits = outputs.logits[0, sep_token_index, :]
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

print(outputs)