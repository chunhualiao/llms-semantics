import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

color_words = ['blue', 'orange', 'red', 'grey', 'white', 'black', 'yellow', 'purple', 'green']

# Path to your JSON file
file_path = 'color_embeddings.json'

# Read the JSON file and convert it back into a Python dictionary
with open(file_path, 'r') as json_file:
    color_embeddings = json.load(json_file)

# Now, loaded_color_embeddings is a Python dictionary with the same content as was saved
print(color_embeddings['red'])

# Convert lists to PyTorch tensors for computation
for color in color_words:
    color_embeddings[color] = torch.tensor(color_embeddings[color])
  
# Compute cosine similarities
similarity_matrix = torch.zeros((len(color_words), len(color_words)))

for i, color1 in enumerate(color_words):
    for j, color2 in enumerate(color_words):
        similarity_matrix[i, j] = torch.nn.functional.cosine_similarity(
            color_embeddings[color1].unsqueeze(0),
            color_embeddings[color2].unsqueeze(0)
        )

# Visualize the similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=True, xticklabels=color_words, yticklabels=color_words, cmap='coolwarm', fmt=".2f")
plt.title('Pairwise Cosine Similarities between Color Embeddings')
# Save the plot to a file in PDF format
plt.savefig('color_embeddings_similarity_matrix.pdf', dpi=300, bbox_inches='tight')

plt.show()
