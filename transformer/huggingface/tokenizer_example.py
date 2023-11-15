from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch import nn


model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
# print(encoding)

pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
print(pt_batch)


pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
pt_outputs = pt_model(**pt_batch)
print(pt_outputs)
print("--------------")


pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
print(pt_predictions)
