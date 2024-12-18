import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

gene1 = 'ANK3'
gene2 = 'SIRPA'


def load_and_get_slice(filepath):
    """Load nifti file and return a middle slice"""
    img = nib.load(filepath)
    data = img.get_fdata()
    # Get middle slice from the third dimension
    middle_slice = data[:, :, data.shape[2]//2]
    return middle_slice

def create_visualization():
    # File paths
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

    # Create figure with custom gridspec to accommodate colorbar and row labels
    fig = plt.figure(figsize=(20, 8))
    # Add more space on the left for gene names
    gs = plt.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.05])
    
    # Initialize vmin and vmax for consistent color scaling
    vmin, vmax = float('inf'), float('-inf')
    
    # First pass to find global min and max
    for gene in [gene1, gene2]:
        for method in ['inr', 'inrs', 'inrs_avg', 'abagen']:
            try:
                data = load_and_get_slice(files[gene][method])
                vmin = min(vmin, np.nanmin(data))
                vmax = max(vmax, np.nanmax(data))
            except Exception:
                continue

    # Plot each image
    all_imgs = []
    for row, gene in enumerate([gene1, gene2]):
        # Add gene name text in a larger, bold font
        fig.text(0.08, 0.75 - row * 0.45, gene, 
                fontsize=14, fontweight='bold', 
                horizontalalignment='right',
                verticalalignment='center')
        
        for col, method in enumerate(['inr', 'inrs', 'inrs_avg', 'abagen']):
            ax = fig.add_subplot(gs[row, col])
            try:
                data = load_and_get_slice(files[gene][method])
                
                # Plot the slice with consistent vmin and vmax
                im = ax.imshow(data.T, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                all_imgs.append(im)
                ax.axis('off')
                
                # Add method name for each subplot
                if row == 0:
                    ax.set_title(f'{method}', fontsize=12)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading\n{files[gene][method]}',
                       ha='center', va='center')
                ax.axis('off')

    # Add a single colorbar on the right
    cax = fig.add_subplot(gs[:, -1])
    plt.colorbar(all_imgs[0], cax=cax, label='Expression Level')

    # Add main title
    fig.suptitle(f'Brain Image Visualization: {gene1} vs {gene2} across different methods', fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0.1, 0.03, 0.97, 0.95])
    
    # Save figure
    plt.savefig('brain_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_visualization()