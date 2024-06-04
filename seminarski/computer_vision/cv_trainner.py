
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

def evaluate_model(model, tokenizer, dataset):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    correct = 0

    for example in dataset:
        # Tokenize the text and prepare the inputs
        inputs = tokenizer(example['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Get the prediction
        prediction = torch.argmax(outputs.logits, dim=-1).item()

        # Compare with the label
        correct += (prediction == example['label'])

    accuracy = correct / len(dataset)
    return accuracy


from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification
from torch.utils.data import DataLoader

# Load the cifar10 dataset
dataset = load_dataset("cifar10")

train_dataset = dataset["train"].shuffle(seed=42).select(range(500))  # Use 500 examples for fine-tuning
test_dataset = dataset["test"].shuffle(seed=42).select(range(500))  # Select 500 examples for testing

def transform(example_batch):
    # inputs = image_processor([x for x in example_batch['img']], return_tensors='pt')
    print(example_batch)
    inputs = image_processor([x for x in example_batch['img']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs

train_dataset = train_dataset.with_transform(transform)
test_dataset = test_dataset.with_transform(transform)

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=8)

image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=10)
model.to(device)


# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy='steps',        # Evaluate every `eval_steps`
    load_best_model_at_end=True,         # Load the best model at the end of training
    logging_strategy='steps',
    report_to="tensorboard"
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
torch.cuda.empty_cache()

# Fine-tune the model
trainer.train()
# Evaluate the model after fine-tuning
# accuracy_after = evaluate_model(model, tokenizer, test_dataset)
# print(f"Accuracy after fine-tuning: {accuracy_after:.2f}")


# model_save_path = "./bert_imdb_auto_trainer_500_epoch3"
# model.base_model.save_pretrained(model_save_path)
# tokenizer.save_pretrained(model_save_path)