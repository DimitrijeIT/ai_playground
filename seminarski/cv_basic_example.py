import cv2
import skimage
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers.image_utils import ImageFeatureExtractionMixin
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch 

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
from transformers import AutoModel, AutoProcessor, AutoImageProcessor
mode_name = "google/owlvit-base-patch32"
model = AutoModel.from_pretrained(mode_name)
processor = AutoProcessor.from_pretrained(mode_name)
# processor = AutoImageProcessor.from_pretrained(mode_name)
# model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
# processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")


# Download sample image
image = skimage.data.astronaut()
image = Image.fromarray(np.uint8(image)).convert("RGB")

# plt.figure("Original image")
# plt.imshow(image)
# plt.show()

# Text queries to search the image for
text_queries = ["human face", "rocket", "nasa badge", "star-spangled banner"]


# Process image and text inputs
inputs = processor(text=text_queries, images=image, return_tensors="pt").to(device)

# Print input names and shapes
for key, val in inputs.items():
    print(f"{key}: {val.shape}")

# Set model in evaluation mode
model = model.to(device)
model.eval()

# Get predictions
with torch.no_grad():
  outputs = model(**inputs)
  
# print(outputs)
for k, val in outputs.items():
    if k not in {"text_model_output", "vision_model_output"}:
        print(f"{k}: shape of {val.shape}")

print("\nText model outputs")
for k, val in outputs.text_model_output.items():
    print(f"{k}: shape of {val.shape}")

print("\nVision model outputs")
for k, val in outputs.vision_model_output.items():
    print(f"{k}: shape of {val.shape}") 




mixin = ImageFeatureExtractionMixin()

# Load example image
image_size = model.config.vision_config.image_size
image = mixin.resize(image, image_size)
input_image = np.asarray(image).astype(np.float32) / 255.0

# from transformers import OwlViTFeatureExtractor

# a = OwlViTFeatureExtractor(outputs, image_size)

# print(a)

# Threshold to eliminate low probability predictions
score_threshold = 0.1

# Get prediction logits
logits = torch.max(outputs["logits"][0], dim=-1)
print('\n LOGITS !!!!!!!!! \n')
print(logits)
scores = torch.sigmoid(logits.values).cpu().detach().numpy()

# Get prediction labels and boundary boxes
labels = logits.indices.cpu().detach().numpy()
boxes = outputs["pred_boxes"][0].cpu().detach().numpy()

def plot_predictions(input_image, text_queries, scores, boxes, labels):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()

    for score, box, label in zip(scores, boxes, labels):
      if score < score_threshold:
        continue

      cx, cy, w, h = box
      ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
              [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
      ax.text(
          cx - w / 2,
          cy + h / 2 + 0.015,
          f"{text_queries[label]}: {score:1.2f}",
          ha="left",
          va="top",
          color="red",
          bbox={
              "facecolor": "white",
              "edgecolor": "red",
              "boxstyle": "square,pad=.3"
          })
    
plot_predictions(input_image, text_queries, scores, boxes, labels)
plt.show()
