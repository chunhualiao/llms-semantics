import json
import numpy as np

from sklearn.decomposition import PCA

def identify_important_dimensions_with_pca(color_embeddings, n_components=None):
    """
    Use PCA to identify the most important dimensions in the embeddings.
    
    Parameters:
    - color_embeddings: A dictionary with color names as keys and embeddings as values (as numpy arrays or lists).
    - n_components: The number of principal components to compute. If None, all components are computed.
    
    Returns:
    - pca: The PCA object after fitting to the data.
    - important_dimensions: A list of tuples (component_index, explained_variance_ratio)
      sorted by explained variance ratio in descending order.
    """
    # Convert the embeddings to a numpy matrix
    embeddings_matrix = np.array(list(color_embeddings.values()))
    
    # Initialize PCA
    pca = PCA(n_components=n_components)
    
    # Fit PCA on the embeddings matrix
    pca.fit(embeddings_matrix)
    
    # Get the explained variance ratio for each component
    explained_variance_ratios = pca.explained_variance_ratio_
    
    # Sort the components by explained variance ratio in descending order
    sorted_indices = np.argsort(explained_variance_ratios)[::-1]
    sorted_explained_variance_ratios = explained_variance_ratios[sorted_indices]
    
    # Create a list of tuples (component_index, explained_variance_ratio)
    important_dimensions = [(index, variance_ratio) for index, variance_ratio in zip(sorted_indices, sorted_explained_variance_ratios)]
    
    return pca, important_dimensions


def identify_important_dimensions_with_pca_and_show_formulas(color_embeddings, top_n_components=5):
    """
    Use PCA to identify the most important dimensions in the embeddings, focusing on the top N components,
    and show the linear formula for each of these components.
    
    Parameters:
    - color_embeddings: A dictionary with color names as keys and embeddings as values (as numpy arrays or lists).
    - top_n_components: The number of top components for which to show the linear formulas.
    
    Returns:
    - pca: The PCA object after fitting to the data.
    """
    # Convert the embeddings to a numpy matrix
    embeddings_matrix = np.array(list(color_embeddings.values()))
    
    # Initialize and fit PCA
    pca = PCA(n_components=top_n_components)
    pca.fit(embeddings_matrix)
    
    # For each of the top N components, print the linear formula
    for i, component in enumerate(pca.components_):
        formula = " + ".join([f"({coef:.3f} * dim_{idx})" for idx, coef in enumerate(component)])
        print(f"Component {i}: {formula}")
    
    return pca

def analyze_embedding_dimensions(color_embeddings):
    """
    Analyze which dimensions in the embeddings contribute the most to the representation of colors
    by calculating the variance of each dimension across the set of color embeddings.

    Parameters:
    - color_embeddings: A dictionary with color names as keys and embeddings as values (as numpy arrays or lists).
    
    Returns:
    - A sorted list of tuples (dimension_index, variance) from highest to lowest variance.
    """
    # Convert the embeddings to a numpy matrix for easier analysis
    embeddings_matrix = np.array(list(color_embeddings.values()))
    
    # Calculate the variance of each dimension
    variances = np.var(embeddings_matrix, axis=0)
    
    # Sort the dimensions by variance in descending order
    sorted_indices = np.argsort(variances)[::-1]
    sorted_variances = variances[sorted_indices]
    
    # Create a sorted list of tuples (dimension_index, variance)
    sorted_dimensions_by_variance = [(index, variance) for index, variance in zip(sorted_indices, sorted_variances)]
    
    return sorted_dimensions_by_variance

color_words = ['blue', 'orange', 'red', 'grey', 'white', 'black', 'yellow', 'purple', 'green']

# Path to your JSON file
file_path = 'color_embeddings.json'

# Read the JSON file and convert it back into a Python dictionary
with open(file_path, 'r') as json_file:
    color_embeddings = json.load(json_file)

# Now, loaded_color_embeddings is a Python dictionary with the same content as was saved
#print(color_embeddings['blue'])

# Analyze the embedding dimensions
sorted_dimensions_by_variance = analyze_embedding_dimensions(color_embeddings)

# Print the top 10 dimensions with the highest variance
print("Top 10 dimensions by variance:")
for dim, var in sorted_dimensions_by_variance[:10]:
    print(f"Dimension {dim} with variance {var:.4f}")

# Identify the most important dimensions with PCA
pca, important_dimensions = identify_important_dimensions_with_pca(color_embeddings)

# Print the top components by explained variance ratio
print("Top components by explained variance ratio:")
for component_index, variance_ratio in important_dimensions[:5]:
    print(f"Component {component_index}: Explained variance ratio = {variance_ratio:.4f}")  

# Identify the most important dimensions and show formulas for the top components
pca = identify_important_dimensions_with_pca_and_show_formulas(color_embeddings)
