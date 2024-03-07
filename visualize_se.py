# %%
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

# Load your data
matter = "white"  # or "grey"
df = pd.read_csv("./data/se.csv").sort_values(by='se')

def get_slice(filename):
    img = nib.load(filename)
    data = img.get_fdata()
    return data[:, :, data.shape[2] // 2]

images = []

for i, row in df.iterrows():
    id = row['gene_symbol']
    order_val = row['se']
    
    slice1 = get_slice(f"nii_results_sep/{matter}_{id}.nii.gz")
    slice2 = get_slice(f"nii_results_full/{matter}_{id}.nii.gz")
    
    plt.close('all')
    
    # Create a new figure for plotting
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    canvas = FigureCanvas(fig)

    axs[0].imshow(slice1.T, cmap="gray", origin="lower")
    axs[0].set_title(f"Saperate Gene Trained")
    axs[0].axis('off')
    
    axs[1].imshow(slice2.T, cmap="gray", origin="lower")
    axs[1].set_title(f"Ordered Genes Trained Together")
    axs[1].axis('off')
    
    fig.suptitle(f"Gene {i+1}/{df.shape[0]} {id} - {matter} matter - se: {order_val:.5f}")
    
    # Save the figure to a buffer
    canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.show()
    plt.close(fig)  # Close the figure to free memory
    
    time.sleep(0.05)  # Adjust the time as needed
    clear_output(wait=True)
    
    images.append(image)

imageio.mimsave(f'{matter}_matter_genes.gif', images, duration=1)

# %%
