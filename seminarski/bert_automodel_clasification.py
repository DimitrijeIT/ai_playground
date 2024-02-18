from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(hidden_size, num_labels)
    
    def forward(self, features):
        # You might want to use the [CLS] token's features or apply pooling to the features
        # Here, we use the [CLS] token's features (features[:, 0, :]) as input to the classifier
        logits = self.dense(features[:, 0, :])
        return logits

# Initialize the classification head
hidden_size = model.config.hidden_size  # Typically 768 for BERT-base
num_labels = 2  # For binary classification
classification_head = ClassificationHead(hidden_size, num_labels)

# Example text
# texts = ["This is an example sentence", "Another example"]
texts = ["I feel very angry", "I am very happy"]
# Tokenize and encode the texts
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Forward pass through the base model
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]

# Forward pass through the classification head
logits = classification_head(last_hidden_states)  # Shape: [batch_size, num_labels]

# Optionally, apply softmax to get probabilities
probabilities = torch.softmax(logits, dim=-1)

print(probabilities)
