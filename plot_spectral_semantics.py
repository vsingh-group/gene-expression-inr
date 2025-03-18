import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import MinMaxScaler

# Create output directory if it doesn't exist
os.makedirs("./manuscript_imgs/combined", exist_ok=True)

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
    """Compute spectral embedding and return all intermediate matrices and eigendata"""
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
    
    # Return all matrices and eigen information
    return X, affinity, D, L, L_norm, eigenvalues, eigenvectors[:, 1:n_components+1]

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

def plot_eigenvalues(eigenvalues, ax):
    """Plot eigenvalues"""
    # Create x values starting from 1
    x_values = range(1, len(eigenvalues[1:]) + 1)
    
    # Plot using these x values
    ax.plot(x_values, eigenvalues[1:], 'o-', markersize=4)
    ax.set_xlabel("Eigenvalue Index", fontsize=13)
    ax.set_ylabel("Eigenvalue", fontsize=13)
    
    # Highlight first few eigenvalues (now at positions 1 and 2)
    ax.plot([1, 2], eigenvalues[1:3], 'ro', markersize=5)
    
    # The rest of your function remains the same
    gap_index = 2
    arrow_start = (10.5, 0.75)
    text_position = (100, 0.8)
    
    ax.annotate("Spectral Gap", 
              xy=arrow_start,
              xytext=text_position,
              arrowprops=dict(arrowstyle="->", color="black"))
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title("Eigenvalue Spectral", fontsize=16)

def plot_eigenvectors(eigenvectors, ax):
    """Plot scatter of first two eigenvectors with fixed aspect ratio"""
    # Adjust point size and create scatter plot
    scatter = ax.scatter(eigenvectors[:, 0], eigenvectors[:, 1], 
             cmap='viridis', s=30, alpha=0.7)
    
    ax.set_xlabel("Eigenvector 1", fontsize=13)
    ax.set_ylabel("Eigenvector 2", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Top Eigenvectors Scatter", fontsize=16)
        
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

def create_combined_visualization(donor="9861", matter="83"):
    """Create a combined 3x2 grid visualization with semantic flow chart and spectral embedding analysis"""
    # Load gene data
    gene_df, gene_names = load_gene_data(donor, matter)
    
    # For better visualization in flow chart, use a subset
    n_samples = min(gene_df.shape[1], 20)  # Limit to 20 samples max
    n_genes = min(len(gene_df), n_samples)  # Use same number of genes as samples for square plots
    gene_df_subset = gene_df.iloc[:n_genes, :n_samples]
    
    # Compute spectral embedding and get all data
    X, affinity, D, L, L_norm, eigenvalues, eigenvectors = compute_spectral_embedding(gene_df_subset)
    
    # Try to get correlation matrices
    try:
        unsorted_matrix = get_correlation_matrix(donor, sort=False)
        sorted_matrix = get_correlation_matrix(donor, sort=True)
    except Exception as e:
        print(f"Could not load correlation matrices: {e}")
        # Create dummy matrices if files not found
        unsorted_matrix = np.random.rand(50, 50)
        sorted_matrix = np.random.rand(50, 50)
    
    # Create a larger figure with a 3x2 grid
    fig = plt.figure(figsize=(16, 20))  # Increased height to allow for more space
    
    # Create GridSpec with 3 rows and 2 columns
    # Modified height_ratios to increase spacing between rows 2 and 3
    # Added a fourth row with a small height to create space between rows 2 and 3
    gs = gridspec.GridSpec(4, 2, 
                         height_ratios=[0.7, 1, 0.2, 1.12],  # Added a 0.2 height row as a spacer
                         width_ratios=[1, 1], 
                         wspace=0.05, 
                         hspace=0.1)  # Increased hspace for more vertical spacing
    
    # === ROW 1: Semantic Flow Chart (spans 2 columns) ===
    # Use a nested GridSpec for the flow chart to create 5 equal columns
    gs_flow = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0, :], wspace=0.1)
    
    # Plot gene expression data
    ax1 = fig.add_subplot(gs_flow[0])
    # Use more samples to make the plot more square
    num_samples = min(20, gene_df.shape[1])
    samples_to_show = gene_df.iloc[:, :num_samples].T
    if samples_to_show.shape[0] < samples_to_show.shape[1]:
        genes_to_show = min(samples_to_show.shape[0], samples_to_show.shape[1])
        samples_to_show = samples_to_show.iloc[:, :genes_to_show]
    
    sns.heatmap(samples_to_show, cmap="viridis", ax=ax1, 
                cbar_kws={"orientation": "horizontal", "shrink": 0.8, "pad": 0.06})
    ax1.set_title("Gene Expression", fontsize=14)
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal')
    
    # Add connecting arrow
    ax1.annotate("", xy=(1.12, 0.5), xytext=(1, 0.5), 
                 xycoords=ax1.transAxes, textcoords=ax1.transAxes,
                 arrowprops=dict(arrowstyle="->", color="black", lw=2))
    
    # Plot affinity matrix
    ax2 = fig.add_subplot(gs_flow[1])
    sns.heatmap(affinity, cmap="Blues", ax=ax2, 
                cbar_kws={"orientation": "horizontal", "shrink": 0.8, "pad": 0.06})
    ax2.set_title("Affinity Matrix", fontsize=14)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_aspect('equal')
    
    # Add connecting arrow
    ax2.annotate("", xy=(1.12, 0.5), xytext=(1, 0.5), 
                 xycoords=ax2.transAxes, textcoords=ax2.transAxes,
                 arrowprops=dict(arrowstyle="->", color="black", lw=2))
    
    # Plot degree matrix D
    ax3 = fig.add_subplot(gs_flow[2])
    sns.heatmap(D, cmap="Greens", ax=ax3, 
                cbar_kws={"orientation": "horizontal", "shrink": 0.8, "pad": 0.06})
    ax3.set_title("Degree Matrix", fontsize=14)
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_aspect('equal')
    
    # Add connecting arrow
    ax3.annotate("", xy=(1.12, 0.5), xytext=(1, 0.5), 
                 xycoords=ax3.transAxes, textcoords=ax3.transAxes,
                 arrowprops=dict(arrowstyle="->", color="black", lw=2))
    
    # Plot Laplacian matrix L
    ax4 = fig.add_subplot(gs_flow[3])
    sns.heatmap(L, cmap="Reds", ax=ax4, 
                cbar_kws={"orientation": "horizontal", "shrink": 0.8, "pad": 0.06})
    ax4.set_title("Laplacian Matrix", fontsize=14)
    ax4.set_xlabel("")
    ax4.set_ylabel("")
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_aspect('equal')
    
    # Add connecting arrow
    ax4.annotate("", xy=(1.12, 0.5), xytext=(1, 0.5), 
                 xycoords=ax4.transAxes, textcoords=ax4.transAxes,
                 arrowprops=dict(arrowstyle="->", color="black", lw=2))
    
    # Plot normalized Laplacian matrix
    ax5 = fig.add_subplot(gs_flow[4])
    sns.heatmap(L_norm, cmap="PuRd", ax=ax5, 
                cbar_kws={"orientation": "horizontal", "shrink": 0.8, "pad": 0.06})
    ax5.set_title("Normalized Laplacian", fontsize=14)
    ax5.set_xlabel("")
    ax5.set_ylabel("")
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.set_aspect('equal')
    
    # Add flow chart title
    plt.figtext(0.5, 0.88, f"Spectral Embedding Semantic Flow", 
                fontsize=16, ha='center')
    
    # === ROW 2-3: Spectral Embedding Analysis (2x2 grid) ===
    _, _, _, _, _, eigenvalues, eigenvectors = compute_spectral_embedding(gene_df)
    
    # B) Eigenvalue plot (row 2, col 1)
    ax_eigenvalues = fig.add_subplot(gs[1, 0])
    plot_eigenvalues(eigenvalues, ax_eigenvalues)
    ax_eigenvalues.set_aspect('auto')
    
    # C) Eigenvectors scatter plot (row 2, col 2)
    ax_eigenvectors = fig.add_subplot(gs[1, 1])
    plot_eigenvectors(eigenvectors, ax_eigenvectors)
        
    # D) Unsorted correlation heatmap (row 4, col 1) - Changed from row 3 to row 4 due to spacer row
    ax_unsorted = fig.add_subplot(gs[3, 0])
    
    # Handle labels for correlation matrices
    num_genes = len(unsorted_matrix)
    max_ticks = 30  # Maximum number of tick labels to show
    
    if num_genes > max_ticks:
        # Calculate interval to display labels
        interval = max(1, num_genes // max_ticks)
        gene_indices = range(0, num_genes, interval)
        
        # Create labels
        if hasattr(unsorted_matrix, 'index'):
            gene_labels = [unsorted_matrix.index[i] if i < len(unsorted_matrix.index) else "" for i in gene_indices]
        else:
            gene_labels = [str(i) for i in gene_indices]
        
        # Create label mask
        x_mask = np.zeros(num_genes, dtype=bool)
        y_mask = np.zeros(num_genes, dtype=bool)
        for i in gene_indices:
            if i < num_genes:
                x_mask[i] = True
                y_mask[i] = True
                
        # Create tick positions
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
    
    # Create unsorted heatmap
    sns.heatmap(unsorted_matrix, ax=ax_unsorted, 
                annot=False, cmap='coolwarm', 
                linewidths=0.5, square=True,
                xticklabels=False, yticklabels=False,
                cbar=False)
    
    # Set selective tick positions and labels
    ax_unsorted.set_xticks(x_ticks + 0.5)
    ax_unsorted.set_xticklabels([gene_labels[i] for i in range(len(x_ticks))], rotation=90, size=8)
    ax_unsorted.set_yticks(y_ticks + 0.5)
    ax_unsorted.set_yticklabels([gene_labels[i] for i in range(len(y_ticks))], size=8)
    
    ax_unsorted.set_title(f"Unsorted Gene Correlation Matrix", fontsize=16)
    ax_unsorted.xaxis.tick_top()
    ax_unsorted.xaxis.set_label_position('top')
    ax_unsorted.tick_params(left=True, bottom=False)
    ax_unsorted.set_ylabel('')
    
    # E) Sorted correlation heatmap (row 4, col 2) - Changed from row 3 to row 4 due to spacer row
    ax_sorted = fig.add_subplot(gs[3, 1])
    
    # Get labels for sorted matrix
    if num_genes > max_ticks:
        if hasattr(sorted_matrix, 'index'):
            sorted_labels = [sorted_matrix.index[i] if i < len(sorted_matrix.index) else "" for i in gene_indices]
        else:
            sorted_labels = [str(i) for i in gene_indices]
    else:
        if hasattr(sorted_matrix, 'index'):
            sorted_labels = list(sorted_matrix.index)
        else:
            sorted_labels = [str(i) for i in range(num_genes)]
    
    # Create sorted heatmap
    sns.heatmap(sorted_matrix, ax=ax_sorted, 
                annot=False, cmap='coolwarm', 
                linewidths=0.5, square=True,
                xticklabels=False, yticklabels=False,
                cbar=False)
    
    # Set tick positions and labels
    ax_sorted.set_xticks(x_ticks + 0.5)
    ax_sorted.set_xticklabels([sorted_labels[i] for i in range(len(x_ticks))], rotation=90, size=8)
    ax_sorted.set_yticks(y_ticks + 0.5)
    ax_sorted.set_yticklabels([sorted_labels[i] for i in range(len(y_ticks))], size=8)
    
    ax_sorted.set_title(f"Sorted Gene Correlation Matrix", fontsize=16)
    ax_sorted.xaxis.tick_top()
    ax_sorted.xaxis.set_label_position('top')
    ax_sorted.tick_params(left=True, bottom=False)
    ax_sorted.set_ylabel('')


    # *** DIRECTLY ADJUST THE SIZE OF THE SECOND ROW PLOTS ***
    # Get the current positions
    pos_eigenvalues = ax_eigenvalues.get_position()
    pos_eigenvectors = ax_eigenvectors.get_position()

    # Calculate the width reduction (reduce width by 20%)
    width_reduction = 0.1

    # Calculate new positions with reduced width
    new_width_eigenvalues = pos_eigenvalues.width * (1 - width_reduction)
    new_width_eigenvectors = pos_eigenvectors.width * (1 - width_reduction)

    # Create new bounding boxes
    new_pos_eigenvalues = [
        pos_eigenvalues.x0 + (pos_eigenvalues.width * width_reduction/2),  # Adjust x to keep centered
        pos_eigenvalues.y0,  # Top stays the same
        new_width_eigenvalues,  # Reduced width
        pos_eigenvalues.height  # Height stays the same
    ]

    new_pos_eigenvectors = [
        pos_eigenvectors.x0 + (pos_eigenvectors.width * width_reduction/2),  # Adjust x to keep centered
        pos_eigenvectors.y0,  # Top stays the same
        new_width_eigenvectors,  # Reduced width
        pos_eigenvectors.height  # Height stays the same
    ]

    # Apply new positions
    ax_eigenvalues.set_position(new_pos_eigenvalues)
    ax_eigenvectors.set_position(new_pos_eigenvectors)

    # *** DIRECTLY ADJUST THE SIZE OF THE THIRD ROW PLOTS (CORRELATION MATRICES) ***
    # Get the current positions
    pos_unsorted = ax_unsorted.get_position()
    pos_sorted = ax_sorted.get_position()

    # Calculate the size increase (increase by 8%)
    increase_factor = 0.08

    # Calculate new positions with increased size
    new_width_unsorted = pos_unsorted.width * (1 + increase_factor)
    new_height_unsorted = pos_unsorted.height * (1 + increase_factor)
    new_width_sorted = pos_sorted.width * (1 + increase_factor)
    new_height_sorted = pos_sorted.height * (1 + increase_factor)

    # Create new bounding boxes
    new_pos_unsorted = [
        pos_unsorted.x0 - (new_width_unsorted - pos_unsorted.width)/2,  # Adjust x to keep centered
        pos_unsorted.y0 - (new_height_unsorted - pos_unsorted.height)/2,  # Adjust y to keep centered
        new_width_unsorted,  # Increased width
        new_height_unsorted  # Increased height
    ]

    new_pos_sorted = [
        pos_sorted.x0 - (new_width_sorted - pos_sorted.width)/2,  # Adjust x to keep centered
        pos_sorted.y0 - (new_height_sorted - pos_sorted.height)/2,  # Adjust y to keep centered
        new_width_sorted,  # Increased width
        new_height_sorted  # Increased height
    ]

    # Apply new positions
    ax_unsorted.set_position(new_pos_unsorted)
    ax_sorted.set_position(new_pos_sorted)
    
    # Add a common color bar for the heatmaps
    cbar_ax = fig.add_axes([0.14, 0.08, 0.745, 0.008])
    plt.colorbar(ax_sorted.collections[0], cax=cbar_ax, orientation='horizontal')

    # Add labels to the first row (flow chart)
    ax1.text(-0.08, 1.15, 'a', transform=ax1.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')

    # Add labels to the second row
    ax_eigenvalues.text(-0.08, 1.08, 'b', transform=ax_eigenvalues.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')
    ax_eigenvectors.text(-0.08, 1.08, 'c', transform=ax_eigenvectors.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')

    # Add labels to the third row
    ax_unsorted.text(-0.08, 1.08, 'd', transform=ax_unsorted.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')
    ax_sorted.text(-0.08, 1.08, 'e', transform=ax_sorted.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    
    # Save the figure
    output_file = f"./manuscript_imgs/combined/combined_spectral_visualization_donor_{donor}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(f"./manuscript_imgs/combined/combined_spectral_visualization_donor_{donor}.pdf", 
                bbox_inches='tight', dpi=300)
    
    print(f"Saved combined visualization to {output_file}")
    plt.close()

def generate_all_visualizations():
    """Generate combined visualizations for both donors"""
    donors = ["9861", "10021"]
    matter = "83"
    
    for donor in donors:
        print(f"Processing donor {donor}...")
        create_combined_visualization(donor, matter)
    
    print("All visualizations generated successfully!")

if __name__ == "__main__":
    # Generate all the combined visualizations
    generate_all_visualizations()