import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import matplotlib
# matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load pre-trained model tokenizer and model from HuggingFace
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize input
input_sentence = "This is a blue chair that Marko is sitting on."
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

# Apply t-SNE to reduce dimensionality for visualization
# Adjust perplexity to be less than the number of samples
perplexity = min(5, initial_embeddings_np.shape[0] - 1)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)

# Concatenate the initial embeddings and encoder outputs for visualization
all_encodings = np.vstack((initial_embeddings_np, encoder_outputs_np))
tsne_results = tsne.fit_transform(all_encodings)

# Separate the t-SNE results back into initial embeddings and encoder outputs
tsne_initial = tsne_results[:initial_embeddings_np.shape[0], :]
tsne_encoder = tsne_results[initial_embeddings_np.shape[0]:, :]

# Plot the t-SNE results
plt.figure(figsize=(12, 8))
plt.scatter(tsne_initial[:, 0], tsne_initial[:, 1], color='blue', label='Initial Embeddings')
plt.scatter(tsne_encoder[:, 0], tsne_encoder[:, 1], color='red', label='Last Hidden Layer Outputs')

# Annotate the points with the corresponding tokens
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].tolist())
for i, token in enumerate(tokens):
    plt.annotate(token, (tsne_initial[i, 0], tsne_initial[i, 1]), fontsize=9, color='blue')
    plt.annotate(token, (tsne_encoder[i, 0], tsne_encoder[i, 1]), fontsize=9, color='red')

plt.title('t-SNE Visualization of BERT Embeddings and Last Hidden Layer Outputs')
plt.legend()
plt.show()
