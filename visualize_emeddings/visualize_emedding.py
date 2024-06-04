import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load pre-trained model tokenizer and model from HuggingFace
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize input
input_sentence = "Transformers are amazing for NLP tasks."
inputs = tokenizer(input_sentence, return_tensors='pt')

# Get initial token embeddings
initial_embeddings = model.embeddings.word_embeddings(inputs['input_ids'])

# Get encoder stack outputs
with torch.no_grad():
    outputs = model(**inputs)

# Get the last hidden state (encoder stack output)
encoder_outputs = outputs.last_hidden_state

# Prepare data for visualization
initial_embeddings_np = initial_embeddings.squeeze().detach().numpy()
encoder_outputs_np = encoder_outputs.squeeze().detach().numpy()

# Concatenate the initial embeddings and encoder outputs for visualization
all_encodings = np.vstack((initial_embeddings_np, encoder_outputs_np))

# Apply t-SNE to reduce dimensionality for visualization
# Adjust perplexity to be less than the number of samples
perplexity = min(5, all_encodings.shape[0] - 1)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
tsne_results = tsne.fit_transform(all_encodings)

# Separate the t-SNE results back into initial embeddings and encoder outputs
tsne_initial = tsne_results[:initial_embeddings_np.shape[0], :]
tsne_encoder = tsne_results[initial_embeddings_np.shape[0]:, :]

# Plot the t-SNE results
plt.figure(figsize=(10, 6))
plt.scatter(tsne_initial[:, 0], tsne_initial[:, 1], color='blue', label='Initial Embeddings')
plt.scatter(tsne_encoder[:, 0], tsne_encoder[:, 1], color='red', label='Encoder Outputs')
for i in range(tsne_initial.shape[0]):
    plt.annotate(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][i].item()), 
                 (tsne_initial[i, 0], tsne_initial[i, 1]), fontsize=9)
plt.title('t-SNE Visualization of BERT Encodings')
plt.legend()
plt.show()
