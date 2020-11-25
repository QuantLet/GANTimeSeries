import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

img = Image.open('vg.jpg').convert('RGB')
pix_val = list(img.getdata())
pix_array = np.array(pix_val)
pix_array = pix_array.reshape((768, 970, 3))

sns.heatmap(pix_array[:,:,0],cmap='Reds')
plt.xticks([])
plt.yticks([])
plt.savefig('vg_reds.png', transparent=True)
plt.close()

sns.heatmap(pix_array[:, :, 1], cmap='Blues')
plt.xticks([])
plt.yticks([])
plt.savefig('vg_blues.png', transparent=True)
plt.close()

sns.heatmap(pix_array[:, :, 2], cmap='Greens')
plt.xticks([])
plt.yticks([])
plt.savefig('vg_greens.png', transparent=True)
plt.close()