from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AdamW, AutoModel
import torch

# Load the IMDb dataset
dataset = load_dataset("imdb")
# print(dataset["train"])
# print(dataset["train"][0])

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# # Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

train_dataset = tokenized_datasets["train"]
# test_dataset = tokenized_datasets["test"]

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# # test_loader = DataLoader(test_dataset, batch_size=8)
# print("----------------")
# print(train_dataset[0])

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# model = AutoModel.from_pretrained('bert-base-uncased', num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
a = 0
for epoch in range(1):  # Training for 1 epoch for simplicity
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        model.zero_grad()

        # outputs = model(**batch)
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['label'])
        
        # outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        # print("----------------")
        # for s in batch['input_ids']:
        #     print(s)
        #     print("----------------")
        #     print(tokenizer.convert_ids_to_tokens(s))
        #     print("----------------")
        # print(outputs)
        # print("----------------")
        # print(outputs.logits)
        # print(batch['label'])
        # print("----------------")
        
        loss = outputs.loss
        print(loss)
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        
        print("\n \n \n")
        a = a + 1
        if a > 3:
            break

    # # Evaluation step (optional)
    # model.eval()
    # total_eval_accuracy = 0
    # for batch in test_loader:
    #     batch = {k: v.to(device) for k, v in batch.items()}

    #     with torch.no_grad():
    #         outputs = model(**batch)

    #     logits = outputs.logits
    #     predictions = torch.argmax(logits, dim=-1)
    #     total_eval_accuracy += torch.sum(predictions == batch["labels"]).item()

    # avg_val_accuracy = total_eval_accuracy / len(test_dataset)
    # print(f"Accuracy: {avg_val_accuracy:.4f}")


# SAVING MODEL
# model.save_pretrained('/path/to/save/model')
# tokenizer.save_pretrained('/path/to/save/tokenizer')
