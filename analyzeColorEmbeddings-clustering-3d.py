import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Visualization
    # Visualize the clusters in 3D space
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each point and add a text label
    for i, (x, y, z) in enumerate(reduced_embeddings):
        ax.scatter(x, y, z, color=plt.cm.viridis(labels[i] / n_clusters))
        ax.text(x, y, z, '%s' % (color_names[i]), size=10, zorder=1, color='k')
    
    ax.set_title('Clustering of Color Embeddings in 3D Space with Labels')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    
    plt.savefig('clustering-color-embeddings-3d.pdf', dpi=300, bbox_inches='tight')
    plt.show()

color_words = ['blue', 'orange', 'red', 'grey', 'white', 'black', 'yellow', 'purple', 'green']

# Path to your JSON file
file_path = 'color_embeddings.json'

# Read the JSON file and convert it back into a Python dictionary
with open(file_path, 'r') as json_file:
    color_embeddings = json.load(json_file)

cluster_and_visualize_colors(color_embeddings, n_clusters=3)

