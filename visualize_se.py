# %%
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Load your data
matter = "white"  # or "grey"
df = pd.read_csv("./data/se.csv").sort_values(by='se')

def get_slice(filename):
    img = nib.load(filename)
    data = img.get_fdata()
    return data[:, :, data.shape[2] // 2]

def compare_images(imageA, imageB):
    data_range = max(imageA.max(), imageB.max()) - min(imageA.min(), imageB.min())
    s = ssim(imageA, imageB, data_range=data_range, multichannel=True)
    return s

images = []
prev_slice1, prev_slice2 = None, None

for i, row in df.iterrows():
    id = row['gene_symbol']
    order_val = row['se']
    
    slice1 = get_slice(f"nii_results_sep/{matter}_{id}.nii.gz")
    slice2 = get_slice(f"nii_results_full/{matter}_{id}.nii.gz")
    
    similarity1, similarity2 = None, None
    if prev_slice1 is not None and prev_slice2 is not None:
        similarity1 = compare_images(prev_slice1, slice1)
        similarity2 = compare_images(prev_slice2, slice2)
    
    prev_slice1, prev_slice2 = np.copy(slice1), np.copy(slice2)

    plt.close('all')
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    canvas = FigureCanvas(fig)

    axs[0].imshow(slice1.T, cmap="gray", origin="lower")
    title1 = f"Separate Gene Trained"
    if similarity1 is not None:
        title1 += f"\nSimilarity: {similarity1:.5f}"
    axs[0].set_title(title1, fontsize=10)
    axs[0].axis('off')
    
    axs[1].imshow(slice2.T, cmap="gray", origin="lower")
    title2 = f"Ordered Genes Trained Together"
    if similarity2 is not None:
        title2 += f"\nSimilarity: {similarity2:.5f}"
    axs[1].set_title(title2, fontsize=10)
    axs[1].axis('off')
    
    fig.suptitle(f"Gene {i+1}/{df.shape[0]} {id} - {matter} matter - se: {order_val:.5f}\n",
                 fontsize=10)
    
    canvas.draw()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.show()
    plt.close(fig)
    
    time.sleep(0.05)
    clear_output(wait=True)
    
    images.append(image)

imageio.mimsave(f'{matter}_matter_genes.gif', images, duration=1)

# %%
