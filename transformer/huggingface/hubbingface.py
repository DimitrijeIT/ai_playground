from transformers import pipeline

classifier = pipeline("sentiment-analysis")
# classifier("We are very happy to show you the 🤗 Transformers library.")

results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
    
# genertor = pipeline("text-generation", model="distilgpt2")
# results = genertor(
#     "In this course I will teach you how to", 
#     max_length=30,
#     num_return_sequences=2,
# )

# for r in results:
    # print(f"label: {r['label']} , score {r['score']}")
