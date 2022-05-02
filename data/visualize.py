from matplotlib.cm import get_cmap
import numpy as np
from PIL import Image

def colorise(input, cmap, vmin=None, vmax=None):

    if isinstance(cmap, str):
        cmap = get_cmap(cmap)

    vmin = float(np.min(input)) if vmin is None else vmin
    vmax = float(np.max(input)) if vmax is None else vmax

    input = (input - vmin) / (vmax - vmin)
    return cmap(input)[..., :3] # :3 -> 0, 1, 2