from datasets import load_dataset
from transformers import AutoModel,AutoTokenizer, AdamW
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a subset of the IMDb dataset
dataset = load_dataset("imdb")
train_dataset = dataset["train"].shuffle(seed=42).select(range(500))  # Select 1,000 examples for training
test_dataset = dataset["test"].shuffle(seed=42).select(range(100))  # Select 1,000 examples for testing

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

train_loader = DataLoader(tokenized_train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(tokenized_test_dataset, batch_size=8)

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(ClassificationHead, self).__init__()
        self.dense = nn.Linear(hidden_size, num_labels)
    
    def forward(self, features):
        logits = self.dense(features[:, 0, :])  # Use the [CLS] token's features
        return logits

class CustomBERTModel(nn.Module):
    def __init__(self, base_model_name, num_labels):
        super(CustomBERTModel, self).__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.classification_head = ClassificationHead(self.base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        logits = self.classification_head(last_hidden_states)
        return logits

model = CustomBERTModel('bert-base-uncased', num_labels=2)

# ENSURE THAT TRANSFORMER IS NOT TRAINED
for param in model.base_model.parameters():
    param.requires_grad = False


def evaluate(model, dataloader):
    model.eval()
    total, correct = 0, 0

    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        labels = inputs.pop("label").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        print(f"Batch evaluate")
    return correct / total

# accuracy_before = evaluate(model, test_loader)
# print(f"Custom classifier: Accuracy before training: {accuracy_before:.2f}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):  # Train for 3 epochs
    model.train()
    for batch in train_loader:
        # inputs = {k: v.to(device) for k, v in batch.items()}
        # labels = inputs.pop("labels").to(device)
        
        # Move each tensor in the batch to the device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Unpack inputs and labels from the batch
        inputs = {k: v for k, v in batch.items() if k != 'label'}
        labels = batch['label']
        
        optimizer.zero_grad()
        
        outputs = model(**inputs)
        
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        print(loss)
        
    print(f"Epoch {epoch}, Loss: {loss.item()}")

accuracy_after = evaluate(model, test_loader)
print(f"Custom classifier: Accuracy after training: {accuracy_after:.2f}")

model_save_path = "./bert_imdb_custom_classifier"
model.base_model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
