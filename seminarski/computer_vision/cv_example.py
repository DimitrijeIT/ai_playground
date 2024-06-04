from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CIFAR-10 dataset from Hugging Face datasets
dataset = load_dataset("cifar10")

train_dataset = dataset["train"].shuffle(seed=42).select(range(500))  # Select 500 examples for training
test_dataset = dataset["test"].shuffle(seed=42).select(range(500))  # Select 500 examples for testing


image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

def transform(example_batch):
    inputs = image_processor([x for x in example_batch['img']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs

# transformed_dataset = dataset.map(transform)
train_dataset = train_dataset.with_transform(transform)
test_dataset = test_dataset.with_transform(transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)



model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=10)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-5)

def evaluate(model, dataloader):
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            # inputs, labels = batch['img'].to(device), batch['label'].to(device)
            # outputs = model(inputs)
            outputs = model(pixel_values = inputs['pixel_values'])
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

accuracy = evaluate(model, test_loader)
print(f"Test Accuracy: {accuracy:.2f}")