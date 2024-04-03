import seaborn as sns
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage.metrics import structural_similarity as ssim
import numpy as np

matter = "white"  # or "grey"
df = pd.read_csv("./data/se_merged.csv")

def get_slice(filename):
    img = nib.load(filename)
    data = img.get_fdata()
    return data[:, :, data.shape[2] // 2]

def compare_images(imageA, imageB):
    data_range = max(imageA.max(), imageB.max()) - min(imageA.min(), imageB.min())
    s = ssim(imageA, imageB, data_range=data_range, multichannel=True)
    return s

grouped_df = df.groupby(['gene_symbol', 'se']).agg(list).reset_index()
grouped_df = grouped_df[['gene_symbol', 'se']]
grouped_df = grouped_df.sort_values(by='se', ascending=True)

n_genes = len(grouped_df)
similarity_matrix = np.zeros((n_genes, n_genes))
genes = grouped_df['gene_symbol'].values

for i, genei in enumerate(genes):
    for j, genej in enumerate(genes):
        slicei = get_slice(f"nii_results_sep/{matter}_{genei}.nii.gz")
        slicej = get_slice(f"nii_results_sep/{matter}_{genej}.nii.gz")
        similarity = compare_images(slicei, slicej)
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity
            
df_for_covariance = pd.DataFrame(similarity_matrix, index=genes, columns=genes)

def draw(covariance_matrix, name):
    plt.figure(figsize=(10, 8))  # You can adjust the figure size as needed
    ax = sns.heatmap(covariance_matrix, 
                annot=False,     # Set to True if you want to see the numbers
                cmap='coolwarm', # Color map
                linewidths=.5)   # Line widths between cells

    plt.title(f'{name} Brain Matrix Heatmap')

    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.xaxis.set_label_position('top')

    plt.savefig(f'{name}_brain_matrix_heatmap.png', dpi=300, bbox_inches='tight')

draw(df_for_covariance.corr(), "Correlation")
draw(df_for_covariance.cov(), "Covariance")

