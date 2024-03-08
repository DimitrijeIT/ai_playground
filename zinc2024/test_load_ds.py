from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="deepchange_dataset", split="train")
print(dataset[0]["label"])