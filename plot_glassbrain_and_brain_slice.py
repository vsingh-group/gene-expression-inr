import os
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn import plotting, image
from sklearn.manifold import SpectralEmbedding
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Set the style parameters
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
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

###### FUNCTIONS FOR GRAPH A (LEFT PANEL) ######

def plot_glassbrain_subplot_a(nii_file, ax, title=None, vmin=None, vmax=None):
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

def create_brain_visualization_a(matter, top_genes, n_genes=5, fig=None, gs=None):
    """
    Create a figure with NxM subplots for different genes and methods (vertical layout)
    with a shared colorbar for all plots - modified to include method labels at top
    
    Parameters:
    -----------
    matter : str
        Matter identifier for file paths
    top_genes : list
        List of gene names to display
    n_genes : int, optional (default=5)
        Number of genes to display
    fig : matplotlib figure, optional
        Figure to draw on. If None, creates a new figure
    gs : GridSpecFromSubplotSpec, optional
        GridSpec to place the plot in
    """
    # Ensure n_genes doesn't exceed available genes
    n_genes = min(n_genes, len(top_genes))
    selected_genes = top_genes[:n_genes]
    
    methods = ['inrs', 'inravg', 'abagen']
    method_labels = ['INR +\nSpectral Embedding', 'INR +\nSpectral Embedding +\nRegional Averaged', 'Abagen']
    
    # Create gridspec within the provided gridspec
    if gs is not None:
        # Use 4 columns: narrow leftmost for gene labels, 3 for plots
        # Add a row at the top for method labels
        subgs = GridSpecFromSubplotSpec(n_genes + 1, 4, gs, 
                                       width_ratios=[0.05, 1, 1, 1],  # Narrow leftmost column
                                       height_ratios=[0.1] + [1] * n_genes,  # First row for method labels
                                       wspace=0.3)
    else:
        # If no gridspec provided, create a new figure
        fig_height = 3 * n_genes + 0.5 + 0.3  # Additional space for colorbar and method labels
        fig = plt.figure(figsize=(10, fig_height))
        subgs = GridSpec(n_genes + 1, 4, figure=fig, 
                         width_ratios=[0.05, 1, 1, 1],  # Narrow leftmost column
                         height_ratios=[0.1] + [1] * n_genes,  # First row for method labels
                         wspace=0.3)
    
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
    
    # Add method labels at the top row
    for col, method_label in enumerate(method_labels):
        label_ax = fig.add_subplot(subgs[0, col+1])  # +1 to skip the first column
        label_ax.axis('off')
        label_ax.text(0.5, 0.5, method_label, 
                     fontsize=14, fontweight='bold',
                     horizontalalignment='center',
                     verticalalignment='center')
    
    # Empty top-left corner cell - ADD LABEL 'a' HERE
    empty_ax = fig.add_subplot(subgs[0, 0])
    empty_ax.axis('off')
    empty_ax.text(0.0, 1.0, 'a', fontsize=24, fontweight='bold', transform=empty_ax.transAxes, va='center', ha='center')
    
    # Second pass: create plots with consistent color scaling
    for row, gene in enumerate(selected_genes):
        # Add gene label on the left side with vertical text - in dedicated column
        label_ax = fig.add_subplot(subgs[row+1, 0])  # +1 for the method labels row
        label_ax.axis('off')
        label_ax.text(1.0, 0.5, gene,  # Position text at right edge of cell
                     va='center', ha='center', fontsize=14, fontweight='bold', rotation=90)
        
        # Create the three method plots
        for col, method in enumerate(methods):
            # Use columns 1, 2, 3 for the actual plots
            ax = fig.add_subplot(subgs[row+1, col+1])  # +1 for the method labels row
            
            if method == 'abagen':
                nii_file = f'result_ibf_2full+mirror_TT/nii_abagen/{gene}_{matter}_abagen.nii.gz'
            elif method == 'inravg':
                nii_file = f'result_ibf_2full+mirror_TT/nii_inravg/{gene}_{matter}_inrs.nii.gz'
            else:  # inrs
                nii_file = f'result_ibf_2full+mirror_TT/nii_inrs/{gene}_{matter}_inrs.nii.gz.nii.gz'
            
            title = None  # No need for title since we have labels on the edges
            result = plot_glassbrain_subplot_a(nii_file, ax, title, vmin=global_vmin, vmax=global_vmax)
            
            if result is not None and result[0] is not None:
                displays_by_col[col].append(result[0])
    
    # Add a single shared colorbar for all plots
    if all_unique_vals and gs is None:
        cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02])  # [left, bottom, width, height]
        norm = plt.Normalize(vmin=global_vmin, vmax=global_vmax)
        sm = plt.cm.ScalarMappable(cmap=sns.color_palette("rocket_r", as_cmap=True), norm=norm)
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Expression Level', fontsize=16)
    
    return fig, global_vmin, global_vmax

def get_top_genes(matter, n_genes=5):
    """
    Get top N genes based on spectral embedding
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
    
    return top_genes

###### FUNCTIONS FOR GRAPH B (RIGHT PANEL) ######

def plot_glassbrain_subplot_b(nii_file, ax, title=None, vmin=None, vmax=None):
    """Plot glass brain visualization in a specific subplot"""
    if not os.path.exists(nii_file):
        ax.text(0.5, 0.5, 'File not found', ha='center', va='center')
        ax.axis('off')
        return None, None, None
        
    try:
        nii_img = nib.load(nii_file)
        volume_data = nii_img.get_fdata()
        unique_vals = np.unique(volume_data)
        unique_vals = unique_vals[unique_vals != 0.]
        
        plot_vmin = vmin if vmin is not None else (unique_vals.min() if len(unique_vals) > 0 else 0)
        plot_vmax = vmax if vmax is not None else (unique_vals.max() if len(unique_vals) > 0 else 1)
        
        display = plotting.plot_glass_brain(
            nii_img,
            display_mode='z',
            colorbar=False,
            plot_abs=False,
            cmap=sns.color_palette("rocket_r", as_cmap=True),
            threshold=0,
            vmin=plot_vmin,
            vmax=plot_vmax,
            alpha=0.6,
            annotate=False,
            axes=ax
        )
        
        if title:
            ax.set_title(title, fontsize=14, pad=2, x=0.5)
            
        return display, unique_vals.min() if len(unique_vals) > 0 else 0, unique_vals.max() if len(unique_vals) > 0 else 1
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
        ax.axis('off')
        return None, None, None

def create_visualization_b(gene1="ANK3", gene2="SIRPA", fig=None, gs=None):
    """
    Create brain visualization for two genes
    
    Parameters:
    -----------
    gene1 : str
        Name of first gene
    gene2 : str
        Name of second gene
    fig : matplotlib figure, optional
        Figure to draw on. If None, creates a new figure
    gs : GridSpecFromSubplotSpec, optional
        GridSpec to place the plot in
    """
    # Define methods and their labels - reordered to match the example image
    methods = ['inr', 'inrs', 'inrs_avg', 'abagen']
    method_labels = ['INR', 'INR +\nSpectral Embedding', 'INR +\nSpectral Embedding +\nRegional Averaged', 'Abagen']
    
    files = {
        gene1: {
            'inr': f'/home/xizheng/brain-inr/nii_9861_83_new_sep/{gene1}_83_new_inr.nii.gz',
            'inrs': f'/home/xizheng/brain-inr/result_ibf_2full+mirror_TT/nii_9861_83_new/{gene1}_83_new_inrs.nii.gz',
            'inrs_avg': f'/home/xizheng/brain-inr/result_ibf_2full+mirror_TT/nii_9861_83_new/{gene1}_83_new_inrs_avg.nii.gz',
            'abagen': f'/home/xizheng/brain-inr/result_ibf_2full+mirror_TT/nii_abagen/{gene1}_83_new_abagen.nii.gz',
        },
        gene2: {
            'inr': f'/home/xizheng/brain-inr/nii_9861_83_new_sep/{gene2}_83_new_inr.nii.gz',
            'inrs': f'/home/xizheng/brain-inr/result_ibf_2full+mirror_TT/nii_9861_83_new/{gene2}_83_new_inrs.nii.gz',
            'inrs_avg': f'/home/xizheng/brain-inr/result_ibf_2full+mirror_TT/nii_9861_83_new/{gene2}_83_new_inrs_avg.nii.gz',
            'abagen': f'/home/xizheng/brain-inr/result_ibf_2full+mirror_TT/nii_abagen/{gene2}_83_new_abagen.nii.gz',
        }
    }

    # Create figure if not provided
    if fig is None and gs is None:
        fig = plt.figure(figsize=(8, 12))
        gs = GridSpec(5, 3,
                     width_ratios=[0.1, 1, 1],
                     height_ratios=[0.1, 1, 1, 1, 1],
                     hspace=0.15,
                     wspace=0.01)
    
    # Create subgridspec if provided a larger gridspec
    if gs is not None:
        subgs = GridSpecFromSubplotSpec(5, 3, gs,
                                       width_ratios=[0.05, 1, 1],
                                       height_ratios=[0.1, 1, 1, 1, 1],
                                       hspace=0.15,
                                       wspace=0.01)  # Keep this value - we'll adjust manually later
    else:
        subgs = gs
    
    # Add 'b' label to the top-left corner cell
    empty_corner_ax = fig.add_subplot(subgs[0, 0])
    empty_corner_ax.axis('off')
    empty_corner_ax.text(0.0, 1.0, 'b', fontsize=24, fontweight='bold', transform=empty_corner_ax.transAxes, va='center', ha='center')

    # First pass to find global min and max
    global_vmin = float('inf')
    global_vmax = float('-inf')
    
    # Add gene names as column headers
    for col, gene in enumerate([gene1, gene2]):
        label_ax = fig.add_subplot(subgs[0, col+1])
        label_ax.axis('off')
        label_ax.text(0.5, 0.5, gene,
                     fontsize=14, fontweight='bold',
                     horizontalalignment='center',
                     verticalalignment='center')

    # Add method labels on the left side
    for row, method_label in enumerate(method_labels):
        label_ax = fig.add_subplot(subgs[row+1, 0])
        label_ax.axis('off')
        label_ax.text(1.0, 0.5, method_label,
                     fontsize=14, fontweight='bold',
                     rotation=90,
                     horizontalalignment='center',
                     verticalalignment='center')
    
    # Loop by method (row) and gene (column)
    for row, method in enumerate(methods):
        for col, gene in enumerate([gene1, gene2]):
            ax = fig.add_subplot(subgs[row+1, col+1])
            display, vmin, vmax = plot_glassbrain_subplot_b(
                files[gene][method], 
                ax
            )
            if vmin is not None and vmax is not None:
                global_vmin = min(global_vmin, vmin)
                global_vmax = max(global_vmax, vmax)

    # Plot brain images by method (row) and gene (column) with consistent min/max
    for row, method in enumerate(methods):
        for col, gene in enumerate([gene1, gene2]):
            ax = fig.add_subplot(subgs[row+1, col+1])
            display, _, _ = plot_glassbrain_subplot_b(
                files[gene][method], 
                ax,
                vmin=global_vmin,
                vmax=global_vmax
            )

    return fig, global_vmin, global_vmax

###### MERGED VISUALIZATION FUNCTION ######

def create_merged_visualization(matter="83_new", n_genes=5, gene1="ANK3", gene2="SIRPA", 
                               output_path="./manuscript_imgs"):
    """
    Create a merged visualization with both brain visualizations
    
    Parameters:
    -----------
    matter : str
        Matter identifier for file paths
    n_genes : int
        Number of top genes to display in first visualization
    gene1 : str
        Name of first gene for second visualization
    gene2 : str
        Name of second gene for second visualization
    output_path : str
        Directory path where output files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Get top genes for left panel
    top_genes = get_top_genes(matter, n_genes)
    
    # Create figure with proper size ratio
    fig = plt.figure(figsize=(18, 12))  # Wider to accommodate both panels
    
    # Create main gridspec with two columns (one for each visualization)
    main_gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.7], wspace=0.1)
    
    # Create subplot gridspecs with reduced top margin
    left_gs = GridSpecFromSubplotSpec(1, 1, main_gs[0, 0])
    right_gs = GridSpecFromSubplotSpec(1, 1, main_gs[0, 1])
    
    # Adjust figure margins to reduce white space
    fig.subplots_adjust(top=0.95, bottom=0.12)
    
    # REMOVE THESE LINES since we're now adding the labels directly in the subplots
    # fig.text(0.12, 0.95, "a", fontsize=24, fontweight='bold')
    # fig.text(0.62, 0.95, "b", fontsize=24, fontweight='bold')
    
    # Create left panel (A) - the method labels are now handled inside the function
    fig, vmin_a, vmax_a = create_brain_visualization_a(matter, top_genes, n_genes, fig, left_gs[0, 0])
    
    # Create right panel (B)
    fig, vmin_b, vmax_b = create_visualization_b(gene1, gene2, fig, right_gs[0, 0])
    
    # Add colorbars appropriate for each panel - adjusted to match example image
    # Colorbar for left panel
    cbar_ax_a = fig.add_axes([0.17, 0.07, 0.38, 0.02])  # [left, bottom, width, height]
    norm_a = plt.Normalize(vmin=0.0, vmax=1.0)  # Fixed range as shown in example
    sm_a = plt.cm.ScalarMappable(cmap=sns.color_palette("rocket_r", as_cmap=True), norm=norm_a)
    cbar_a = plt.colorbar(sm_a, cax=cbar_ax_a, orientation='horizontal')
    cbar_a.ax.tick_params(labelsize=12)

    # Colorbar for right panel
    cbar_ax_b = fig.add_axes([0.63, 0.07, 0.25, 0.02])  # [left, bottom, width, height]
    norm_b = plt.Normalize(vmin=-0.2, vmax=1.0)  # Fixed range as shown in example
    sm_b = plt.cm.ScalarMappable(cmap=sns.color_palette("rocket_r", as_cmap=True), norm=norm_b)
    cbar_b = plt.colorbar(sm_b, cax=cbar_ax_b, orientation='horizontal')
    cbar_b.ax.tick_params(labelsize=12)

    # Save the merged visualization
    plt.savefig(f'{output_path}/merged_brain_visualization.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f'{output_path}/merged_brain_visualization.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Merged visualization saved to {output_path}/merged_brain_visualization.pdf")

if __name__ == "__main__":
    create_merged_visualization(
        matter="83_new",
        n_genes=5,
        gene1="ANK3",
        gene2="SIRPA",
        output_path="./manuscript_imgs"
    )