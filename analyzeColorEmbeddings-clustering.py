import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np



def cluster_and_visualize_colors(color_embeddings, n_clusters=5):
    """
    Perform K-means clustering on color embeddings and visualize the groupings.
    
    Parameters:
    - color_embeddings: A dictionary with color names as keys and embeddings as values.
    - n_clusters: Number of clusters to form.
    """
    # Extract embeddings and color names
    embeddings = np.array(list(color_embeddings.values()))
    color_names = list(color_embeddings.keys())
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Visualization
    plt.figure(figsize=(10, 8))
    for i in range(n_clusters):
        # Select data belonging to the current cluster
        cluster_indices = np.where(labels == i)[0]
        cluster_embeddings = reduced_embeddings[cluster_indices]
        cluster_names = [color_names[idx] for idx in cluster_indices]
        
        # Plot
        plt.scatter(cluster_embeddings[:, 0], cluster_embeddings[:, 1], label=f'Cluster {i+1}')
        
        # Annotate points with color names
        for point, name in zip(cluster_embeddings, cluster_names):
            plt.text(point[0], point[1], name, fontsize=9)
    
    plt.title('Color Embeddings Clustering')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.legend()
    plt.savefig('clustering-color-embeddings.pdf', dpi=300, bbox_inches='tight')
    plt.show()

color_words = ['blue', 'orange', 'red', 'grey', 'white', 'black', 'yellow', 'purple', 'green']

# Path to your JSON file
file_path = 'color_embeddings.json'

# Read the JSON file and convert it back into a Python dictionary
with open(file_path, 'r') as json_file:
    color_embeddings = json.load(json_file)

cluster_and_visualize_colors(color_embeddings, n_clusters=3)

