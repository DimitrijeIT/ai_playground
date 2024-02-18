import torch
from transformers import AutoModel, AutoTokenizer

text = "Capital of France is [MASK]"

model_ckpt = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

tokenizer =AutoTokenizer.from_pretrained(model_ckpt)
inputs = tokenizer(text, return_tensors="pt")

inputs = {k:v.to(device) for k,v in inputs.items()}

model.eval()

with torch.no_grad(): #Save resources
 outputs = model(**inputs)
 print(outputs)
#  string_tokens = tokenizer.convert_tokens_to_string(outputs.pooler_output[:])
#  print(string_tokens)