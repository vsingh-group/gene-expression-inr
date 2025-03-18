import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
import seaborn as sns

# Set the style
plt.style.use('seaborn-v0_8-paper')

# Update plotting parameters
plt.rcParams.update({
    # 'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 8,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})

def plot_glassbrain_subplot(nii_file, ax, title=None, vmin=None, vmax=None):
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

def create_visualization(output_path="./", gene1="ANK3", gene2="SIRPA"):
    """
    Create brain visualization for two genes
    
    Parameters:
    -----------
    output_path : str
        Directory path where output files will be saved
    gene1 : str
        Name of first gene
    gene2 : str
        Name of second gene
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Define methods and their labels
    methods = ['inr', 'inrs', 'inrs_avg', 'abagen']
    method_labels = ['INR One Gene', 'INR Continuous', 'INR Averaged', 'Abagen']
    
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

    # Create figure with adjusted height
    fig = plt.figure(figsize=(12, 6))
    
    # Create GridSpec with minimal vertical spacing
    gs = plt.GridSpec(2, 5, 
                     width_ratios=[0.1, 1, 1, 1, 1], 
                     hspace=0.0,
                     wspace=0.01,
                     height_ratios=[1, 1])

    # First pass to find global min and max
    global_vmin = float('inf')
    global_vmax = float('-inf')
    
    for row, gene in enumerate([gene1, gene2]):
        for col, method in enumerate(methods):
            ax = plt.subplot(gs[row, col+1])
            display, vmin, vmax = plot_glassbrain_subplot(
                files[gene][method], 
                ax
            )
            if vmin is not None and vmax is not None:
                global_vmin = min(global_vmin, vmin)
                global_vmax = max(global_vmax, vmax)

    # Second pass to plot with consistent scaling
    for row, gene in enumerate([gene1, gene2]):
        # Add gene labels in their own subplot with vertical orientation
        label_ax = plt.subplot(gs[row, 0])
        label_ax.axis('off')
        label_ax.text(0.5, 0.5, gene,
                     fontsize=14, fontweight='bold',
                     rotation=90,
                     horizontalalignment='center',
                     verticalalignment='center')
        
        for col, (method, label) in enumerate(zip(methods, method_labels)):
            ax = plt.subplot(gs[row, col+1])
            display, _, _ = plot_glassbrain_subplot(
                files[gene][method], 
                ax,
                title=label if row == 0 else None,
                vmin=global_vmin,
                vmax=global_vmax
            )

    # Add colorbar with adjusted position
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=sns.color_palette("rocket_r", as_cmap=True),
                              norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax))
    plt.colorbar(sm, cax=cax, label='Expression Level')

    # Generate output filenames
    base_filename = f"brain_slice"
    png_path = os.path.join(output_path, f"{base_filename}.png")
    pdf_path = os.path.join(output_path, f"{base_filename}.pdf")
    
    # Save both PNG and pdf versions
    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.close()

if __name__ == "__main__":
    # Example usage with custom parameters
    create_visualization(
        output_path="./manuscript_imgs",
    )