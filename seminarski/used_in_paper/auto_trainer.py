
import torch
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


# def evaluate_model(model, tokenizer, dataset):
#     model.eval()
#     correct = 0
#     with torch.no_grad():
#         for example in dataset:
#             inputs = tokenizer(example['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
#             outputs = model(**inputs)
#             prediction = torch.argmax(outputs.logits, dim=-1).item()
#             correct += (prediction == example['label'])
#     accuracy = correct / len(dataset)
#     return accuracy



from datasets import load_dataset
# Load the IMDb dataset
dataset = load_dataset("imdb")
test_dataset = dataset["test"].shuffle(seed=42).select(range(100))  # Select 100 examples for testing


from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# # Evaluate the model without fine-tuning
# accuracy_before = evaluate_model(model, tokenizer, test_dataset)
# print(f"Accuracy before fine-tuning: {accuracy_before:.2f}")

# -------------------------------------------------------------------------------------------------
from transformers import Trainer, TrainingArguments

# Prepare a small subset of the training dataset for fine-tuning
train_dataset = dataset["train"].shuffle(seed=42).select(range(100))  # Use 1000 examples for fine-tuning


from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Function to tokenize the examples
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

# Load the IMDb dataset
dataset = load_dataset("imdb")
small_train_dataset = dataset["train"].shuffle(seed=42).select(range(500))  # Smaller subset for training
# small_test_dataset = dataset["test"].shuffle(seed=42).select(range(100))  # Smaller subset for testing
small_test_dataset = test_dataset

# Tokenize the datasets
tokenized_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = small_test_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])




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
    train_dataset=tokenized_train_dataset,
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