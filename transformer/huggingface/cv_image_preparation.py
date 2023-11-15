from datasets import load_dataset
from transformers import AutoImageProcessor
import numpy as np
import matplotlib.pyplot as plt

dataset = load_dataset("food101", split="train[:100]")

print(dataset[0]["image"])

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

def transforms(examples):
    images = [img.convert("RGB") for img in examples["image"]]
    examples["pixel_values"] = image_processor(images, do_resize=False, return_tensors="pt")["pixel_values"]
    return examples

dataset.set_transform(transforms)

img = dataset[0]["pixel_values"]
plt.imshow(img.permute(1, 2, 0))
