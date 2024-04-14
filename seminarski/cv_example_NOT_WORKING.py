from datasets import load_dataset

# Load CIFAR-10 dataset from Hugging Face datasets
dataset = load_dataset("cifar10")
# print(dataset['train'][0])

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from datasets import Features, ClassLabel, Array3D

# Define the preprocessing transformations
transform = Compose([
    Resize((224, 224)),  # Resize the image to 224x224 pixels
    ToTensor(), # Convert the image to a PyTorch tensor
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize the tensor
])

# Function to apply transformations
def transform_example(example):
    example['img'] = transform(example['img'])
    return example

# Apply transformations to the dataset
# dataset = dataset.with_transform(transform_example)

# Update the dataset format for PyTorch
# dataset.set_format(type='torch', columns=['img', 'label'])


from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTImageProcessor
import torch.nn as nn
from torch.optim import AdamW
import torch

# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
def transform(examples):
    # Apply feature extractor, which handles resizing and normalization
    # return feature_extractor(examples['img'], return_tensors='pt')
    
    # # Apply feature extractor transformations
    transformed = feature_extractor(examples['img'], return_tensors='pt')
    # # Include labels in the transformed output
    # transformed['labels'] = examples['label']
    return transformed
    
    # for image in examples["img"]:
    #     print(image)
    #     break
    
    # images = [feature_extractor(image, return_tensors="pt") for image in examples["img"]]
    # # Extract the tensor from the processed output
    # images = [image["pixel_values"].squeeze() for image in images]
    # examples["img"] = images
    # return examples



# transformed_dataset = dataset.with_transform(transform)

train_dataset = dataset["train"].shuffle(seed=42).select(range(500))  # Select 500 examples for training
test_dataset = dataset["test"].shuffle(seed=42).select(range(500))  # Select 500 examples for testing

# transformed_dataset = dataset.map(transform)
train_dataset = train_dataset.map(transform, remove_columns=["img"])
test_dataset = test_dataset.map(transform, remove_columns=["img"])

# Load a pre-trained ViT model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=10)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-5)

from torch.utils.data import DataLoader

# train_loader = DataLoader(dataset['train'], batch_size=8, shuffle=True)
# test_loader = DataLoader(dataset['test'], batch_size=8)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# for batch in train_loader:
#     print(batch["img"].shape)  # Should output something like torch.Size([8, 3, 224, 224])
#     print(batch["labels"])  # Should output the labels for the batch
#     break  # Just check the first batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(3):  # Example: 3 epochs
    model.train()
    for batch in train_loader:
        # inputs, labels = batch['img'].to(device), batch['label'].to(device)
        # optimizer.zero_grad()
        # print(inputs)
        # outputs = model(inputs)
        
        
        # inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        inputs = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(**inputs)
        
        
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")


def evaluate(model, dataloader):
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch['img'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

accuracy = evaluate(model, test_loader)
print(f"Test Accuracy: {accuracy:.2f}")

