import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def visualize_color_embeddings_3d(color_embeddings):
    """
    Apply t-SNE to reduce the color embeddings to 3 dimensions and visualize in 3D space.

    Parameters:
    - color_embeddings: A dictionary with color names as keys and embeddings as values (as lists or numpy arrays).
    """
    # Extract color names and prepare the embeddings matrix
    color_words = list(color_embeddings.keys())

    # Prepare the embeddings matrix
    embeddings_matrix = np.array([color_embeddings[color] for color in color_words])

    # Apply t-SNE to reduce to 3 dimensions
#    tsne = TSNE(n_components=3, random_state=42)
   # Adjust perplexity for small datasets
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(5, len(color_words)-1))
   
    embeddings_3d = tsne.fit_transform(embeddings_matrix)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for the embeddings
    for i, color in enumerate(color_words):
        ax.scatter(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], label=color)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('3D Visualization of Color Embeddings')
    ax.legend()

    # Save the plot to a file in PDF format
    plt.savefig('3D-visual-color-embeddings_all.pdf', dpi=300, bbox_inches='tight')
    plt.show()


#color_words = ['blue', 'orange', 'red', 'grey', 'white', 'black', 'yellow', 'purple', 'green']
color_words = ['blue', 'orange', 'red', 'grey', 'white', 'black', 'yellow', 'purple', 'green']

# Path to your JSON file
file_path = 'color_embeddings_all.json'

# Read the JSON file and convert it back into a Python dictionary
with open(file_path, 'r') as json_file:
    color_embeddings = json.load(json_file)

# Now, loaded_color_embeddings is a Python dictionary with the same content as was saved
#print(color_embeddings['red'])

visualize_color_embeddings_3d(color_embeddings)

# Convert lists to PyTorch tensors for computation
#for color in color_words:
#    color_embeddings[color] = torch.tensor(color_embeddings[color])
  

