import torch
from PIL import Image
from torchvision import transforms
from transformers import DeepLabV3ForSemanticSegmentation, DeepLabV3Tokenizer

# Load a pre-trained DeepLabV3 model
model = DeepLabV3ForSemanticSegmentation.from_pretrained("etalab-ia/deeplabv3plus-pytorch-detr-resnet50")

# Load the tokenizer
tokenizer = DeepLabV3Tokenizer.from_pretrained("etalab-ia/deeplabv3plus-pytorch-detr-resnet50")

# Load and preprocess the image
image_path = "path_to_your_image.jpg"
image = Image.open(image_path)

# Define transformation for preprocessing
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Preprocess the image
input_image = preprocess(image).unsqueeze(0)

# Perform segmentation
with torch.no_grad():
    outputs = model(input_image)

# Get the predicted segmentation mask
logits = outputs.logits
predicted_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

# You can visualize or save the segmentation mask
predicted_mask_image = Image.fromarray(predicted_mask.astype('uint8'))
predicted_mask_image.save("segmentation_mask.png")

# Optionally, you can apply post-processing to improve the segmentation mask.

print("Segmentation mask saved as segmentation_mask.png")
