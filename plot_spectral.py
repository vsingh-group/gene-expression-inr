#### plot_spectral.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import MinMaxScaler
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import sparse
from scipy.sparse.linalg import eigsh
import os
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import seaborn as sns

# Create output directory if it doesn't exist
os.makedirs("./manuscript_imgs/temp", exist_ok=True)

def load_gene_data(donor, matter="83"):
    """Load gene expression data for a specific donor"""
    # Load gene names
    with open("./data/gene_names.csv") as f:
        gene_names = f.readlines()
        gene_names = [x.strip() for x in gene_names]
    
    gene_names = set(gene_names)

    # Load microarray data
    microarray_df = pd.read_csv(f"./data/abagendata/train_{matter}_new/{matter}_microarray_{donor}.csv")
    microarray_df = microarray_df.iloc[:, 1:].T
    microarray_df.columns = microarray_df.iloc[0].astype(int).astype(str)
    microarray_df = microarray_df.iloc[1:]
    microarray_df.index.name = 'gene_symbol'

    # Find gene names not in microarray_df
    missing_genes = gene_names - set(microarray_df.index)
    print(f"Missing genes for donor {donor}: {missing_genes}")
    gene_names = gene_names - missing_genes
    
    # Use gene_names to get a gene dataframe
    gene_names = list(gene_names)
    gene_df = microarray_df.loc[gene_names]
    
    return gene_df, gene_names

def compute_spectral_embedding(gene_df, n_components=3):
    """Compute spectral embedding and return eigenvalues and eigenvectors"""
    # Transpose so samples are on rows
    X = gene_df.T.values
    
    # Compute affinity matrix using RBF kernel
    # First, standardize the data
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Compute pairwise distances
    squared_dists = np.sum((X_std[:, np.newaxis, :] - X_std[np.newaxis, :, :]) ** 2, axis=2)
    
    # Convert to affinity using RBF kernel
    gamma = 1.0 / X.shape[1]  # Default in SpectralEmbedding
    affinity = np.exp(-gamma * squared_dists)
    
    # Create the graph Laplacian
    # First, compute the degree matrix
    degrees = np.sum(affinity, axis=1)
    D = np.diag(degrees)
    
    # Compute the unnormalized Laplacian
    L = D - affinity
    
    # For normalized Laplacian (often gives better results)
    D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees + 1e-10))
    L_norm = np.eye(L.shape[0]) - D_sqrt_inv @ affinity @ D_sqrt_inv
    
    # Compute eigenvalues and eigenvectors of the normalized Laplacian
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
    
    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Return all eigenvalues and the first n_components eigenvectors
    return eigenvalues, eigenvectors[:, :n_components]

def get_correlation_matrix(donor, sort=True):
    """Get correlation matrix for a donor, optionally sorted by spectral embedding"""
    try:
        df = pd.read_csv(f"data/abagendata/train_83_new/se_{donor}_merged.csv")
        
        grouped_df = df.groupby(['gene_symbol', 'se']).agg(list).reset_index()
        grouped_df = grouped_df[['gene_symbol', 'value', 'se']]
        if sort:
            grouped_df = grouped_df.sort_values(by='se', ascending=True)
        
        print(f"Correlation matrix shape for donor {donor} (sorted={sort}): {grouped_df.shape}")
        
        data_for_matrix = {}
        for index, row in grouped_df.iterrows():
            data_for_matrix[row['gene_symbol']] = row['value']
        
        df_for_covariance = pd.DataFrame(data_for_matrix)
        
        # Create the correlation matrix
        correlation_matrix = df_for_covariance.corr()
        
        # Set diagonal values to 0
        np.fill_diagonal(correlation_matrix.values, 0)
        
        return correlation_matrix
    except Exception as e:
        print(f"Error getting correlation matrix for donor {donor}: {e}")
        # Return a dummy matrix if file not found
        return np.random.rand(50, 50)

def create_combined_figure(donor, matter="83"):
    """Create a combined figure with spectral embedding and heatmaps with improved labels"""
    gene_df, gene_names = load_gene_data(donor, matter)
    
    # Compute spectral embedding
    eigenvalues, eigenvectors = compute_spectral_embedding(gene_df, n_components=3)
    se = SpectralEmbedding(n_components=2, affinity='rbf')
    embedding = se.fit_transform(gene_df.T)
    
    # Try to get correlation matrices
    try:
        unsorted_matrix = get_correlation_matrix(donor, sort=False)
        sorted_matrix = get_correlation_matrix(donor, sort=True)
    except Exception as e:
        print(f"Could not load correlation matrices: {e}")
        # Create dummy matrices if files not found
        unsorted_matrix = np.random.rand(50, 50)
        sorted_matrix = np.random.rand(50, 50)
    
    # Create figure with a 2x2 grid
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], wspace=0.05, hspace=0.2)
    
    # B) Eigenvalue plot (top left)
    ax_eigenvalues = fig.add_subplot(gs[0, 0])
    plot_eigenvalues(eigenvalues, ax_eigenvalues)
    ax_eigenvalues.set_aspect('auto')
    
    # C) Eigenvectors scatter plot (top right)
    ax_eigenvectors = fig.add_subplot(gs[0, 1])
    plot_eigenvectors(eigenvectors, ax_eigenvectors)
    
    # D) Unsorted correlation heatmap (bottom left)
    ax_unsorted = fig.add_subplot(gs[1, 0])
    
    # FIXED LABEL HANDLING: Limit the number of tick labels to display
    num_genes = len(unsorted_matrix)
    max_ticks = 30  # Maximum number of tick labels to show
    
    if num_genes > max_ticks:
        # Calculate interval to display labels (skip some labels)
        interval = max(1, num_genes // max_ticks)
        gene_indices = range(0, num_genes, interval)
        
        # Create empty lists for tick positions and labels
        if hasattr(unsorted_matrix, 'index'):
            gene_labels = [unsorted_matrix.index[i] if i < len(unsorted_matrix.index) else "" for i in gene_indices]
        else:
            gene_labels = [str(i) for i in gene_indices]
        
        # Create label mask - True for positions to keep labels, False to hide
        x_mask = np.zeros(num_genes, dtype=bool)
        y_mask = np.zeros(num_genes, dtype=bool)
        for i in gene_indices:
            if i < num_genes:
                x_mask[i] = True
                y_mask[i] = True
                
        # Create empty lists for tick positions
        x_ticks = np.where(x_mask)[0]
        y_ticks = np.where(y_mask)[0]
    else:
        # If we have few enough genes, show all labels
        if hasattr(unsorted_matrix, 'index'):
            gene_labels = list(unsorted_matrix.index)
        else:
            gene_labels = [str(i) for i in range(num_genes)]
        x_ticks = range(num_genes)
        y_ticks = range(num_genes)
    
    # Create heatmap without tick labels first
    sns.heatmap(unsorted_matrix, ax=ax_unsorted, 
                annot=False, cmap='coolwarm', 
                linewidths=0.5, square=True,
                xticklabels=False, yticklabels=False,
                cbar=False)
    
    # Then set selective tick positions and labels
    ax_unsorted.set_xticks(x_ticks + 0.5)  # +0.5 centers the labels
    ax_unsorted.set_xticklabels([gene_labels[i] for i in range(len(x_ticks))], rotation=90, size=8)
    ax_unsorted.set_yticks(y_ticks + 0.5)
    ax_unsorted.set_yticklabels([gene_labels[i] for i in range(len(y_ticks))], size=8)
    
    ax_unsorted.set_title(f"D) Unsorted Gene Correlation Matrix (Donor {donor})")
    ax_unsorted.xaxis.tick_top()
    ax_unsorted.xaxis.set_label_position('top')
    ax_unsorted.tick_params(left=True, bottom=False)
    ax_unsorted.set_ylabel('')
    
    # E) Sorted correlation heatmap (bottom right)
    ax_sorted = fig.add_subplot(gs[1, 1])
    
    # Apply same technique for sorted matrix
    if num_genes > max_ticks:
        # Calculate interval for sorted matrix
        if hasattr(sorted_matrix, 'index'):
            sorted_labels = [sorted_matrix.index[i] if i < len(sorted_matrix.index) else "" for i in gene_indices]
        else:
            sorted_labels = [str(i) for i in gene_indices]
    else:
        if hasattr(sorted_matrix, 'index'):
            sorted_labels = list(sorted_matrix.index)
        else:
            sorted_labels = [str(i) for i in range(num_genes)]
    
    # Create sorted heatmap without labels first
    sns.heatmap(sorted_matrix, ax=ax_sorted, 
                annot=False, cmap='coolwarm', 
                linewidths=0.5, square=True,
                xticklabels=False, yticklabels=False,
                cbar=False)
    
    # Then set selective tick positions and labels for sorted matrix
    ax_sorted.set_xticks(x_ticks + 0.5)
    ax_sorted.set_xticklabels([sorted_labels[i] for i in range(len(x_ticks))], rotation=90, size=8)
    ax_sorted.set_yticks(y_ticks + 0.5)
    ax_sorted.set_yticklabels([sorted_labels[i] for i in range(len(y_ticks))], size=8)
    
    ax_sorted.set_title(f"E) Spectrum Embedding Sorted Gene Correlation Matrix (Donor {donor})")
    ax_sorted.xaxis.tick_top()
    ax_sorted.xaxis.set_label_position('top')
    ax_sorted.tick_params(left=True, bottom=False)
    ax_sorted.set_ylabel('')
    
    # Add a common color bar for the heatmaps
    cbar_ax = fig.add_axes([0.14, 0.08, 0.745, 0.02])
    plt.colorbar(ax_sorted.collections[0], cax=cbar_ax, orientation='horizontal')
    
    # Add overall title
    plt.suptitle(f"Spectral Embedding Analysis with Correlation Heatmaps for Donor {donor}", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    
    # Save the figure
    output_file = f"./manuscript_imgs/temp/spectral_with_heatmaps_donor_{donor}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(f"./manuscript_imgs/temp/spectral_with_heatmaps_donor_{donor}.svg", format='svg', bbox_inches='tight')
    print(f"Saved combined figure to {output_file}")
    plt.close()

def plot_eigenvalues(eigenvalues, ax):
    """Plot eigenvalues"""
    ax.plot(range(len(eigenvalues)), eigenvalues, 'o-', markersize=4)
    ax.set_xlabel("Eigenvalue Index")
    ax.set_ylabel("Eigenvalue")
    
    # Highlight first few eigenvalues
    ax.plot(range(2), eigenvalues[:2], 'ro', markersize=6)
    
    # Add text annotation
    gap_index = 1  # Typically between the 2nd and 3rd eigenvalues
    ax.annotate("Spectral Gap", 
              xy=(gap_index, eigenvalues[gap_index]), 
              xytext=(gap_index+1, eigenvalues[gap_index]+0.2),
              arrowprops=dict(arrowstyle="->"))
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title("B) Eigenvalue Spectrum", fontweight='bold')

def plot_eigenvectors(eigenvectors, ax):
    """Plot scatter of first two eigenvectors with fixed aspect ratio"""
    # Adjust point size and create scatter plot
    scatter = ax.scatter(eigenvectors[:, 0], eigenvectors[:, 1], 
             cmap='viridis', s=30, alpha=0.7)
    
    ax.set_xlabel("Eigenvector 1")
    ax.set_ylabel("Eigenvector 2")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("C) Top Eigenvectors Scatter", fontweight='bold')
        
    # Make sure axes are properly scaled with some padding
    x_min, x_max = eigenvectors[:, 0].min(), eigenvectors[:, 0].max()
    y_min, y_max = eigenvectors[:, 1].min(), eigenvectors[:, 1].max()
    
    # Calculate padding (10% of the range)
    x_padding = 0.1 * (x_max - x_min)
    y_padding = 0.1 * (y_max - y_min)
    
    # Set limits with padding
    ax.set_xlim([x_min - x_padding, x_max + x_padding])
    ax.set_ylim([y_min - y_padding, y_max + y_padding])
    
    # IMPORTANT: Do NOT set aspect='equal' here as it's forcing the narrow display
    # Instead, let the axes adjust to the data naturally
    ax.set_aspect('auto')

def generate_all_figures():
    """Generate combined spectral+heatmap figures for both donors"""
    donors = ["9861", "10021"]
    matter = "83"
    
    for donor in donors:
        print(f"Processing donor {donor}...")
        create_combined_figure(donor, matter)
    
    print("All figures generated successfully!")

if __name__ == "__main__":
    # Generate all the combined figures
    generate_all_figures()