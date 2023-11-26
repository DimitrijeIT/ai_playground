from datasets import load_dataset
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
import torch   
from torch import nn
from helpers import *
from PIL import Image
import json
import ast
from overlap import *

# ------ Load Data Set and Split into train and test ----------------------
ds = load_dataset("TrainingDataPro/people-tracking-dataset", split="train")

# print(ds[33])
image = ds[33]["image"]
n_array = np.array(image)
original_img = Image.fromarray(n_array)
original_img.save("original.jpeg")

show_image(image, "Original")

anno = ds[33]["annotations"]
# print(type(anno))
# anno = anno.replace("'", '"')
# print("\n Annoration \n " + anno + "\n ----------- \n ")

# anno = np.array(anno)
# for a in anno:
    # print(a['points'])

# j = json.loads(anno)
# print(j["points"])

py_obj = ast.literal_eval(anno)
print("\n \n PY OBJ type : " + str(type(py_obj)) + " \n")
# print(py_obj)
ground_truth = []
for a in py_obj:
    # print(type(a))
    # print(a['points'])
    p1 = a['points'][0]
    x1 = int(p1[0])
    y1 = int(p1[1])
    p2 = a['points'][1]
    x2 = int(p2[0])
    y2 = int(p2[1])
    ground_truth.append((x1,y1,x2,y2))
    # ground_truth.append(a['points'][1])

print("\n ===== Grount truth \n ")
print(ground_truth)

# for p1, p2 in ground_truth:
#     print(str(p1[0]) + " " + str(p1[1]) + " " + str(p2[0]) + " " + str(p2[1]))
for x1,y1,x2,y2 in ground_truth:
    print(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2))
#-------------- pipeline ONLY  --------------------------
segmenter = pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512")
pred_seg = segmenter(image)
print("\n \n ------- PREDICION  \n ")
print(pred_seg)


# show_image(image)
show_image(pred_seg[0]['mask'])
for pred in pred_seg:
    # show_image(pred['mask'], pred['label'])
    if pred['label'] == 'person':
        show_image(pred['mask'], pred['label'])
        n_array = np.array(pred['mask'])
        # 255 where it finds a persion
        print("\n ---- NUMPY ARRAY OF MASK PERSION\n ")
        print(n_array)
        print(type(n_array))
        print(n_array[61:80, 559:600])
        show_image(n_array, "NUMPY ARRAY")
        # img = Image.fromarray(n_array, "RGB")
        img = Image.fromarray(n_array)
        img.show()
        image_filename = "opengenus_image.jpeg"
        img.save(image_filename)

check_bbox_overlap(img, ground_truth)
pred_seg_size = pred_seg[0]['mask'].size
print("Size = ", pred_seg_size)

plt.show(block=False)

input()
plt.close('all')

