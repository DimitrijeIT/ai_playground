from transformers import pipeline
classifier = pipeline("text-classification")
output = classifier("Danas je topao dan")
print(output)

# import torch

# a = torch.cuda.is_available()
# print(a) - FALSE