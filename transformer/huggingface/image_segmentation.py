from datasets import load_dataset

import matplotlib.pyplot as plt
import numpy as np
import torch   

count = 0 
def show_image(img, name = "Image"):
    global count
    plt.figure(name + str(count))
    count = count + 1
    plt.imshow(img)
    # plt.show()


ds = load_dataset("scene_parse_150", split="train[:50]")
ds = ds.train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]
# print(train_ds[0])
# print(train_ds[0]['image'])

# show_image(train_ds[0]['image'])
# show_image(train_ds[100]['image'])
# show_image(train_ds[0]['annotation'])

# def create_label_to_id_and_reverse_mapping():
import json
from huggingface_hub import cached_download, hf_hub_url

repo_id = "huggingface/label-files"
filename = "ade20k-id2label.json"
id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
print("Number of labels", num_labels, id2label[0])
# for i in label2id:
#     print(i)
print(label2id['person']) # 12
# show_image(train_ds[])


# # Preprocessing
from transformers import AutoImageProcessor

checkpoint = "nvidia/mit-b0"
image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=True)

# from torchvision.transforms import ColorJitter

# jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

# def train_transforms(example_batch):
#     images = [jitter(x) for x in example_batch["image"]]
#     labels = [x for x in example_batch["annotation"]]
#     inputs = image_processor(images, labels)
#     return inputs


# def val_transforms(example_batch):
#     images = [x for x in example_batch["image"]]
#     labels = [x for x in example_batch["annotation"]]
#     inputs = image_processor(images, labels)
#     return inputs

# train_ds.set_transform(train_transforms)
# test_ds.set_transform(val_transforms)

# # Evaluate
# import evaluate

# metric = evaluate.load("mean_iou")

# def compute_metrics(eval_pred):
#     with torch.no_grad():
#         logits, labels = eval_pred
#         logits_tensor = torch.from_numpy(logits)
#         logits_tensor = nn.functional.interpolate(
#             logits_tensor,
#             size=labels.shape[-2:],
#             mode="bilinear",
#             align_corners=False,
#         ).argmax(dim=1)

#         pred_labels = logits_tensor.detach().cpu().numpy()
#         metrics = metric.compute(
#             predictions=pred_labels,
#             references=labels,
#             num_labels=num_labels,
#             ignore_index=255,
#             reduce_labels=False,
#         )
#         for key, value in metrics.items():
#             if type(value) is np.ndarray:
#                 metrics[key] = value.tolist()
#         return metrics
    
# Train

from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer

model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)


image = test_ds[0]["image"]

from transformers import pipeline

segmenter = pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512")
# segmenter = pipeline("image-segmentation")
pred_seg = segmenter(image)
print(pred_seg)
# show_image(image)
# # show_image(pred_seg[0]['mask'])
# for pred in pred_seg:
#     show_image(pred['mask'], pred['label'])
pred_seg_size = pred_seg[0]['mask'].size
print("Size = ", pred_seg_size)
# ------------------------------------
# # Visulisation
import matplotlib.pyplot as plt
import numpy as np

# outputs = model(pixel_values=pixel_values)
# logits = outputs.logits.cpu()

# upsampled_logits = nn.functional.interpolate(
#     logits,
#     size=image.size[::-1],
#     mode="bilinear",
#     align_corners=False,
# )
# pred_seg = upsampled_logits.argmax(dim=1)[0]
# # pred_seg = image

ade_palette = np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])

color_seg = np.zeros((pred_seg_size[0], pred_seg_size[1], 3), dtype=np.uint8)
# # color_seg = np.zeros((480, 640, 3), dtype=np.uint8)
# palette = np.array(ade_palette())
palette = ade_palette
for label, color in enumerate(palette):
    color_seg[pred_seg_label == label, :] = color
color_seg = color_seg[..., ::-1]  # convert to BGR

# # plt.figure(figsize=(15, 10))
# plt.figure()
# plt.imshow(color_seg)
# plt.show()

img = np.array(image) * 0.5 + color_seg * 0.5  # plot the image with the segmentation map
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()