import json
import numpy as np

from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial.distance import cosine
import webcolors
from colorsys import rgb_to_hsv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compare_embeddings_with_rgb_and_visualize(color_embeddings):
    """
    Compare color embeddings with human perception represented by RGB values, using webcolors for color name to RGB conversion,
    and visualize the comparisons in 3D space.
    """
    color_to_rgb = {}
    
    # Convert color names to RGB using webcolors
    for color in color_embeddings.keys():
        try:
            rgb = webcolors.name_to_rgb(color)
            color_to_rgb[color] = (rgb.red, rgb.green, rgb.blue)
        except ValueError as e:
            print(f"Warning: {e}. Color '{color}' will be skipped.")
    
    common_colors = list(set(color_embeddings.keys()) & set(color_to_rgb.keys()))
    embeddings_matrix = np.array([color_embeddings[color] for color in common_colors])
    rgb_matrix = np.array([color_to_rgb[color] for color in common_colors])
    
    # Apply PCA to reduce embeddings to 3 dimensions
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings_matrix)
    
    # Normalize RGB values to [0, 1] range for fair comparison
    rgb_normalized = rgb_matrix / 255.0
    
    # Plotting
    fig = plt.figure(figsize=(16, 8))
    
    # Plot reduced embeddings
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], c=rgb_normalized)
    ax1.set_title('Reduced Color Embeddings')
    ax1.set_xlabel('PCA1')
    ax1.set_ylabel('PCA2')
    ax1.set_zlabel('PCA3')
    
    # Plot RGB values
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(rgb_normalized[:, 0], rgb_normalized[:, 1], rgb_normalized[:, 2], c=rgb_normalized)
    ax2.set_title('RGB Values')
    ax2.set_xlabel('R')
    ax2.set_ylabel('G')
    ax2.set_zlabel('B')
    plt.savefig('3D-color-embeddings-RGB.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# main entry
#================================================================
color_words = ['blue', 'orange', 'red', 'grey', 'white', 'black', 'yellow', 'purple', 'green']

# Path to your JSON file
file_path = 'color_embeddings.json'

# Read the JSON file and convert it back into a Python dictionary
with open(file_path, 'r') as json_file:
    color_embeddings = json.load(json_file)

compare_embeddings_with_rgb_and_visualize(color_embeddings)

