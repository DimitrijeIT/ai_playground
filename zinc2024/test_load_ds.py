from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
import torch.nn as nn
from transformers import Trainer, TrainingArguments

# dataset = load_dataset("imagefolder", data_dir="deepchange_dataset", split="train")
# dataset = load_dataset("imagefolder", data_dir="dataset_small", split="train")
# print(dataset[0]["label"])

dataset = load_dataset("imagefolder", data_dir="dataset_small", split="train")
# print(train_dataset[0]["label"])
# print(train_dataset[0])
train_dataset = dataset["train"].shuffle(seed=42).select(range(100))  # Select 500 examples for training
# # test_dataset = dataset["test"].shuffle(seed=42).select(range(500))  # Select 500 examples for testing

image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

def transform(example_batch):
    inputs = image_processor([x for x in example_batch['img']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs

# # # transformed_dataset = dataset.map(transform)
train_dataset = train_dataset.with_transform(transform)
# # # test_dataset = test_dataset.with_transform(transform)

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# # test_loader = DataLoader(test_dataset, batch_size=8)

# from transformers import pipeline
# vision_classifier = pipeline(model="google/vit-base-patch16-224")

# # from PIL import Image
# # import io
# # image = Image.open(io.BytesIO(train_dataset[0][""]))
# preds = vision_classifier(train_dataset[0]["image"])

# print(preds)

from transformers import ViTForImageClassification, AutoModel
model = AutoModel.from_pretrained('google/vit-base-patch16-224-in21k')

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=tokenized_test_dataset,
)
torch.cuda.empty_cache()

# Fine-tune the model
trainer.train()
# Evaluate the model after fine-tuning
accuracy_after = evaluate_model(model, tokenizer, test_dataset)
print(f"Accuracy after fine-tuning: {accuracy_after:.2f}")


# model_save_path = "./bert_imdb_auto_trainer_500_epoch3"
# model.base_model.save_pretrained(model_save_path)
# tokenizer.save_pretrained(model_save_path)