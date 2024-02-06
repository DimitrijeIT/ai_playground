from transformers import pipeline
classifier = pipeline("text-classification")
output = classifier("Danas je topao dan")
print(output)