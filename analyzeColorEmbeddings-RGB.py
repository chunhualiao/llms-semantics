import json
import numpy as np
from sklearn.decomposition import PCA

from scipy.spatial.distance import cosine
import webcolors
from colorsys import rgb_to_hsv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cosine

def safe_normalize(vectors):
    """Normalize vectors avoiding division by zero for zero vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors by setting norms of zero vectors to 1
    norms[norms == 0] = 1
    return vectors / norms


def safe_cosine_similarity(u, v):
    """Compute cosine similarity safely, handling zero vectors."""
    if np.all(u == 0) or np.all(v == 0):
        # Define behavior for zero vectors; here we set similarity to 0
        return 0
    return 1 - cosine(u, v)

def compare_embeddings_with_rgb(color_embeddings):
    """
    Compare color embeddings with human perception represented by RGB values, using webcolors for color name to RGB conversion.
    
    Parameters:
    - color_embeddings: A dictionary with color names as keys and embeddings as values.
    """
    color_to_rgb = {}
    
    # Convert color names to RGB using webcolors
    for color in color_embeddings.keys():
        try:
            rgb = webcolors.name_to_rgb(color)
            color_to_rgb[color] = (rgb.red, rgb.green, rgb.blue)
        except ValueError as e:
            print(f"Warning: {e}. Color '{color}' will be skipped.")
    

    # Similarly, check the RGB values
    for color, rgb in color_to_rgb.items():
        if np.all(np.array(rgb) == 0) or np.isnan(rgb).any():
            print(f"Warning: RGB value for color '{color}' is all zeros or contains NaN.")

    # Ensure we only use colors available in both dictionaries
    common_colors = set(color_embeddings.keys()) & set(color_to_rgb.keys())
    embeddings_matrix = np.array([color_embeddings[color] for color in common_colors])
    rgb_matrix = np.array([color_to_rgb[color] for color in common_colors])
    

    # Normalize both embeddings and RGB values safely
#    embeddings_matrix = safe_normalize(embeddings_matrix)
#    rgb_matrix = safe_normalize(rgb_matrix / 255.0)  # Assuming RGB values are in [0, 255]


    # Apply PCA to reduce embeddings to 3 dimensions
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings_matrix)
    
    # Normalize RGB values to the same scale as reduced embeddings for fair comparison
    # Optionally convert RGB to HSV if needed
    rgb_normalized = rgb_matrix / 255.0
    hsv_matrix = np.array([rgb_to_hsv(*rgb) for rgb in rgb_normalized])
    
    # Compute pairwise cosine similarities within embeddings and RGB/HSV values
    embedding_similarities = np.array([[1 - cosine(u, v) for u in reduced_embeddings] for v in reduced_embeddings])
    rgb_similarities = np.array([[1 - cosine(u, v) for u in hsv_matrix] for v in hsv_matrix])

#    embedding_similarities = np.array([[safe_cosine_similarity(u, v) for u in reduced_embeddings] for v in reduced_embeddings])
#    rgb_similarities = np.array([[safe_cosine_similarity(u, v) for u in hsv_matrix] for v in hsv_matrix])
#    rgb_similarities = np.array([[safe_cosine_similarity(u, v) for u in rgb_matrix] for v in rgb_matrix])
    # Calculate pairwise cosine similarities safely
#    embedding_similarities = np.array([[safe_cosine_similarity(u, v) for u in reduced_embeddings] for v in reduced_embeddings])
#    rgb_similarities = np.array([[safe_cosine_similarity(u, v) for u in rgb_matrix] for v in rgb_matrix])
    
    # Calculate the difference between similarities
    similarity_difference = np.abs(embedding_similarities - rgb_similarities)
    
    # Assess the overall similarity (lower values indicate closer representation)
    overall_similarity = np.mean(similarity_difference)
    
    print(f"Overall similarity between embeddings and RGB: {overall_similarity}")
    
    return overall_similarity

# main entry
#================================================================
color_words = ['blue', 'orange', 'red', 'grey', 'white', 'black', 'yellow', 'purple', 'green']

# Path to your JSON file
file_path = 'color_embeddings_all.json'

# Read the JSON file and convert it back into a Python dictionary
with open(file_path, 'r') as json_file:
    color_embeddings = json.load(json_file)


# Check for zero or NaN vectors in embeddings
for color, embedding in color_embeddings.items():
    if np.all(embedding == 0) or np.isnan(embedding).any():
        print(f"Warning: Embedding for color '{color}' is all zeros or contains NaN.")  

compare_embeddings_with_rgb(color_embeddings)
