from datasets import load_dataset
from transformers import AutoModelForSequenceClassification,AutoTokenizer, AdamW
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

def evaluate(model, dataloader):
    model.eval()
    total, correct = 0, 0

    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        labels = inputs.pop("label").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        outputs = outputs.logits # We need to use logits instead of whole output as previous
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        print(f"Batch evaluate")
    return correct / total


model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# accuracy_before = evaluate(model, test_loader)
# print(f"AutoModelForSequenceClassification: Accuracy before training: {accuracy_before:.2f}")

optimizer = AdamW(model.parameters(), lr=2e-5)
for epoch in range(3):  # Training for 1 epoch for simplicity
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        model.zero_grad()

        # outputs = model(**batch)
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")

accuracy_after = evaluate(model, test_loader)
print(f"AutoModelForSequenceClassification: Accuracy after training: {accuracy_after:.2f}")

model_save_path = "./bert_imdb_fine_tunning"
model.base_model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
