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

def plot_glassbrain_subplot(nii_file, ax, title=None, vmin=None, vmax=None):
    """Plot glass brain visualization in a specific subplot with optional vmin/vmax for consistent color scaling"""
    if not os.path.exists(nii_file):
        ax.text(0.5, 0.5, 'File not found', ha='center', va='center')
        ax.axis('off')
        return
        
    try:
        nii_img = nib.load(nii_file)
        volume_data = nii_img.get_fdata()
        unique_vals = np.unique(volume_data)
        unique_vals = unique_vals[unique_vals != 0.]
        
        # If vmin/vmax not provided, calculate from data
        if vmin is None:
            vmin = unique_vals.min() if len(unique_vals) > 0 else 0
        if vmax is None:
            vmax = unique_vals.max() if len(unique_vals) > 0 else 1
        
        display = plotting.plot_glass_brain(
            nii_img,
            display_mode='l',
            colorbar=False,
            plot_abs=False,
            cmap=sns.color_palette("rocket_r", as_cmap=True),
            threshold=0,
            vmin=vmin,
            vmax=vmax,
            alpha=0.6,
            annotate=False,
            axes=ax
        )
        
        if title:
            ax.set_title(title, fontsize=20, pad=10, x=0.5)
            
        return display, vmin, vmax
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
        ax.axis('off')
        return None, None, None

def create_brain_visualization(matter, top_genes, n_genes=5):
    """
    Create a figure with NxM subplots for different genes and methods (vertical layout)
    with a shared colorbar for all plots
    
    Parameters:
    -----------
    matter : str
        Matter identifier for file paths
    top_genes : list
        List of gene names to display
    n_genes : int, optional (default=5)
        Number of genes to display
    """
    # Ensure n_genes doesn't exceed available genes
    n_genes = min(n_genes, len(top_genes))
    selected_genes = top_genes[:n_genes]
    
    methods = ['abagen', 'inravg', 'inrs']
    method_labels = ['Abagen', 'INR Averaged', 'INR Continuous']
    
    # Adjust figure size - now taller than wide, with space for colorbar
    fig_height = 3 * n_genes + 0.5  # Additional space for colorbar
    fig = plt.figure(figsize=(10, fig_height))
    
    # Create gridspec with genes as rows and methods as columns
    # Add extra space on the left for gene labels
    gs = plt.GridSpec(n_genes, 3, figure=fig, width_ratios=[1]*3, height_ratios=[1]*n_genes, wspace=0.3)
    
    # Add space for gene labels on the left and colorbar at bottom
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    
    # Calculate column positions for method labels
    col_positions = [0.2, 0.5, 0.8]
    
    # Add method labels at the top
    for col, (method, method_label, col_pos) in enumerate(zip(methods, method_labels, col_positions)):
        fig.text(col_pos, 0.98, method_label, 
                 rotation=0, va='center', ha='center', fontsize=18, fontweight='bold')
    
    # First pass: load all NIfTI files to determine global min and max values
    all_unique_vals = []
    
    for row, gene in enumerate(selected_genes):
        for col, method in enumerate(methods):
            if method == 'abagen':
                nii_file = f'result_ibf_2full+mirror_TT/nii_abagen/{gene}_{matter}_abagen.nii.gz'
            elif method == 'inravg':
                nii_file = f'result_ibf_2full+mirror_TT/nii_inravg/{gene}_{matter}_inrs.nii.gz'
            else:  # inrs
                nii_file = f'result_ibf_2full+mirror_TT/nii_inrs/{gene}_{matter}_inrs.nii.gz.nii.gz'
            
            if os.path.exists(nii_file):
                try:
                    nii_img = nib.load(nii_file)
                    volume_data = nii_img.get_fdata()
                    unique_vals = np.unique(volume_data)
                    unique_vals = unique_vals[unique_vals != 0.]
                    if len(unique_vals) > 0:
                        all_unique_vals.extend(unique_vals)
                except Exception:
                    pass
    
    # Determine global min and max values
    global_vmin = min(all_unique_vals) if all_unique_vals else 0
    global_vmax = max(all_unique_vals) if all_unique_vals else 1
    
    displays_by_col = [[] for _ in range(len(methods))]
    
    # Second pass: create plots with consistent color scaling
    for row, gene in enumerate(selected_genes):
        # Add gene label on the left side with vertical text
        fig.text(0.01, 1 - ((row + 0.5) / n_genes), gene, 
                 va='center', ha='center', fontsize=18, fontweight='bold', rotation=90)
        
        for col, method in enumerate(methods):
            ax = fig.add_subplot(gs[row, col])
            
            if method == 'abagen':
                nii_file = f'result_ibf_2full+mirror_TT/nii_abagen/{gene}_{matter}_abagen.nii.gz'
            elif method == 'inravg':
                nii_file = f'result_ibf_2full+mirror_TT/nii_inravg/{gene}_{matter}_inrs.nii.gz'
            else:  # inrs
                nii_file = f'result_ibf_2full+mirror_TT/nii_inrs/{gene}_{matter}_inrs.nii.gz.nii.gz'
            
            title = None  # No need for title since we have labels on the edges
            result = plot_glassbrain_subplot(nii_file, ax, title, vmin=global_vmin, vmax=global_vmax)
            
            if result[0] is not None:
                displays_by_col[col].append(result[0])
    
    # Add a single shared colorbar for all plots
    if all_unique_vals:
        global_vmin = min(all_unique_vals)
        global_vmax = max(all_unique_vals)
        
        # Create single colorbar at the bottom, centered and wider
        cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02])  # [left, bottom, width, height]
        norm = plt.Normalize(vmin=global_vmin, vmax=global_vmax)
        sm = plt.cm.ScalarMappable(cmap=sns.color_palette("rocket_r", as_cmap=True), norm=norm)
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Expression Level', fontsize=16)
    
    # Adjust layout with tighter margins, but leave space for the gene labels on the left
    # and the colorbar at the bottom
    plt.tight_layout(rect=[0.1, 0.15, 0.95, 0.95])
    
    return fig

def visualize_top_genes(matter, n_genes=5, save_path=None):
    """
    Create and save brain visualization for top N genes
    
    Parameters:
    -----------
    matter : str
        Matter identifier for file paths
    n_genes : int, optional (default=5)
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
    fig = create_brain_visualization(matter, top_genes, n_genes=n_genes)

    # Save visualizations
    if save_path is None:
        save_path = 'brain_viz_comparison_vertical'
    
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight', format='pdf')
    plt.close()
    
    return top_genes

# Example usage:
if __name__ == "__main__":
    visualize_top_genes(matter="83_new", n_genes=5, save_path="./manuscript_imgs/fig1_top5_genes_viz_vertical")