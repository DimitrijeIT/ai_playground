from datasets import load_dataset
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
import torch   
from torch import nn
from helpers import *


# ------ Load Data Set and Split into train and test ----------------------
# ds = load_dataset("scene_parse_150", split="train[:50]")
# ds = ds.train_test_split(test_size=0.2)
# train_ds = ds["train"]
# test_ds = ds["test"]

# image = test_ds[0]["image"]

#todo: Find  a good image from dataset to work on

ds = load_dataset("scene_parse_150")
image = ds['train'][45]["image"]
show_image(image, "Original")

#-------------- pipeline ONLY  --------------------------
segmenter = pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512")
pred_seg = segmenter(image)
# print(pred_seg)

# show_image(image)
# show_image(pred_seg[0]['mask'])
# for pred in pred_seg:
#     show_image(pred['mask'], pred['label'])
# pred_seg_size = pred_seg[0]['mask'].size
# print("Size = ", pred_seg_size)

plt.show()