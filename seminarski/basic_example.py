# from transformers import pipeline
# classifier = pipeline("text-classification")
# output = classifier("Today is beautiful day.")
# print(output)

# import torch

# a = torch.cuda.is_available()
# print(a) # FALSE

# from transformers import pipeline
# qa_pipeline = pipeline("question-answering")
# result = qa_pipeline(question="What are sons doing?", context="I have two sons. Older sun is Marko and younger one Nikola. They go to high school.")
# print(result)

from transformers import pipeline
generator = pipeline("text-generation")
result = generator("After a long day")
print(result)

# from transformers import pipeline
# vision_classifier = pipeline(model="google/vit-base-patch16-224")
# preds = vision_classifier(
#     images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
# )
# print(preds)