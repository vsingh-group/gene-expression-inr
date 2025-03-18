matter = "83_new"

import os
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
from nilearn import plotting, image
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding

# Set the style parameters
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    # 'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 16,
    'axes.titlesize': 14,
    'xtick.labelsize': 15,
    'ytick.labelsize': 12,
    'legend.fontsize': 8,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})

def plot_glassbrain_subplot(nii_file, ax, title=None):
    """Plot glass brain visualization in a specific subplot"""
    if not os.path.exists(nii_file):
        ax.text(0.5, 0.5, 'File not found', ha='center', va='center')
        ax.axis('off')
        return
        
    try:
        nii_img = nib.load(nii_file)
        volume_data = nii_img.get_fdata()
        unique_vals = np.unique(volume_data)
        unique_vals = unique_vals[unique_vals != 0.]
        
        display = plotting.plot_glass_brain(
            nii_img,
            display_mode='l',
            colorbar=False,
            plot_abs=False,
            cmap=sns.color_palette("rocket_r", as_cmap=True),
            threshold=0,
            vmin=unique_vals.min() if len(unique_vals) > 0 else 0,
            vmax=unique_vals.max() if len(unique_vals) > 0 else 1,
            alpha=0.6,
            annotate=False,
            axes=ax
        )
        
        if title:
            ax.set_title(title, fontsize=20, pad=10, x=0.5)
            
        return display, unique_vals.min() if len(unique_vals) > 0 else 0, unique_vals.max() if len(unique_vals) > 0 else 1
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
        ax.axis('off')
        return None, None, None

def create_brain_visualization(matter, top_genes, n_cols=8):
    """
    Create a figure with 3xN subplots for different methods and genes
    
    Parameters:
    -----------
    matter : str
        Matter identifier for file paths
    top_genes : list
        List of gene names to display
    n_cols : int, optional (default=8)
        Number of columns (genes) to display
    """
    # Ensure n_cols doesn't exceed available genes
    n_cols = min(n_cols, len(top_genes))
    selected_genes = top_genes[:n_cols]
    
    # Adjust figure size based on number of columns
    fig_width = 3 * n_cols  # 3 inches per column
    fig = plt.figure(figsize=(fig_width, 8))
    
    methods = ['abagen', 'inravg', 'inrs']
    method_labels = ['Abagen', 'INR Averaged', 'INR Continuous']
    
    # Create main gridspec with reduced spacing
    gs = plt.GridSpec(3, n_cols, figure=fig, width_ratios=[1]*n_cols, height_ratios=[1]*3, hspace=0.3)
    
    # Add space for method labels on the left
    fig.subplots_adjust(left=0.05, bottom=0.05)
    
    # Calculate row positions for better alignment
    row_positions = [0.8, 0.5, 0.2]
    
    for row, (method, method_label, row_pos) in enumerate(zip(methods, method_labels, row_positions)):
        displays = []
        vmin_list = []
        vmax_list = []
        
        # Add method label (vertical) aligned with row center
        fig.text(0.02, row_pos, method_label, 
                rotation=90, va='center', ha='center', fontsize=18, fontweight='bold')
        
        for col, gene in enumerate(selected_genes):
            ax = fig.add_subplot(gs[row, col])
            
            if row == 0:
                title = f'{gene}'
            else:
                title = None
                
            if method == 'abagen':
                nii_file = f'result_ibf_2full+mirror_TT/nii_abagen/{gene}_{matter}_abagen.nii.gz'
            elif method == 'inravg':
                nii_file = f'result_ibf_2full+mirror_TT/nii_inravg/{gene}_{matter}_inrs.nii.gz'
            else:  # inrs
                nii_file = f'result_ibf_2full+mirror_TT/nii_inrs/{gene}_{matter}_inrs.nii.gz.nii.gz'
            
            result = plot_glassbrain_subplot(nii_file, ax, title)
            if result[0] is not None:
                displays.append(result[0])
                vmin_list.append(result[1])
                vmax_list.append(result[2])
        
        # Add colorbar with adjusted position
        if displays and vmin_list and vmax_list:
            cbar_ax = fig.add_axes([0.92, row_pos - 0.1, 0.01, 0.2])
            vmin, vmax = min(vmin_list), max(vmax_list)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=sns.color_palette("rocket_r", as_cmap=True), norm=norm)
            plt.colorbar(sm, cax=cbar_ax)
            cbar_ax.tick_params(labelsize=14)
    
    # Adjust layout with tighter margins
    plt.tight_layout(rect=[0.05, 0.05, 0.91, 0.95])
    
    return fig

def visualize_top_genes(matter, n_genes=8, save_path=None):
    """
    Create and save brain visualization for top N genes
    
    Parameters:
    -----------
    matter : str
        Matter identifier for file paths
    n_genes : int, optional (default=8)
        Number of top genes to display
    save_path : str, optional (default=None)
        Base path for saving the visualization. If None, uses 'brain_viz_comparison'
    """
    # Load data and calculate embeddings
    df_inravg = pd.read_csv(f"./data/{matter}_interpolation_inrs.csv", index_col='label')

    # Calculate spectral embedding
    gene_df_embedding = df_inravg.T
    embedding = SpectralEmbedding(n_components=1)
    gene_embedding = embedding.fit_transform(gene_df_embedding)
    gene_df_embedding = pd.DataFrame({"se": gene_embedding[:, 0].flatten()}, index=gene_df_embedding.index)
    gene_df_embedding = gene_df_embedding.sort_values(by="se", ascending=False)

    # Get top N genes
    top_genes = gene_df_embedding.head(n_genes).index.to_list()
    print(f"Top {n_genes} genes by spectral embedding:", top_genes)

    # Create visualization
    fig = create_brain_visualization(matter, top_genes, n_cols=n_genes)

    # Save visualizations
    if save_path is None:
        save_path = 'brain_viz_comparison'
    
    plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', format='pdf')
    plt.close()
    
    return top_genes

# Example usage:
if __name__ == "__main__":
    # To visualize top 8 genes (default)
    visualize_top_genes(matter="83_new", n_genes=8, save_path="./manuscript_imgs/fig1_top8_genes_viz")
    
    visualize_top_genes(matter="83_new", n_genes=5, save_path="./manuscript_imgs/fig1_top5_genes_viz")