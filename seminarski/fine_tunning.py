import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        label = self.labels[idx]
        return inputs, label

texts = ["This course is great", "I did not like this movie"]
labels = [1, 0]  # 1: Positive, 0: Negative

dataset = TextDataset(texts, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# optimizer = AdamW(model.parameters(), lr=1e-5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(2):  # Training for 2 epochs for simplicity
    for inputs, labels in dataloader:
        inputs = {k: v.squeeze().to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        model.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")

